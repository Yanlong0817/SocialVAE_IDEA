"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import repeat
import numpy as np

from models.layer_utils import *
from scipy import interpolate
from copy import deepcopy

from models.social_encoder import Encoder
from PIL import Image
from torchvision import transforms
import clip

# -----------------------------------------------------------------------------
class Final_Model(nn.Module):

    def __init__(
        self,
        dataset_name: str,
        past_len: int,
        future_len: int,
        n_embd: int,
        dropout: float,
        int_num_layers_list = [3, 2],
        n_head: int = 8,
        forward_expansion: int = 2,
        block_size: int = 128,
        vocab_size: int = 2,
        goal_num: int = 20,
        lambda_des: float = 1.0,
        lambda_j: float = 1.0,
        lambda_recon: float = 1.0,
    ):
        super(Final_Model, self).__init__()

        self.name_model = "PPT_Model"
        self.past_len = past_len  # 8
        self.future_len = future_len  # 12
        self.total_len = self.past_len + self.future_len  # 20
        self.dataset_name = dataset_name
        self.lambda_des = lambda_des
        self.goal_num = goal_num
        self.n_embd = n_embd
        self.dropout = dropout
        self.lambda_j = lambda_j
        self.lambda_recon = lambda_recon
        # 位置编码
        self.wpe=nn.Embedding(block_size, n_embd)  # 位置编码

        # 对输入轨迹进行编码
        # self.traj_encoder = nn.Linear(config.vocab_size, config.n_embd)  # FC(2, 128)  *********************
        self.traj_encoder = nn.Sequential(
            nn.Linear(vocab_size, n_embd),
            # nn.ReLU(),
            # nn.Dropout(self.dropout)
        )  # FC(2, 128)
        self.AR_Model = GPT(
            vocab_size=vocab_size,
            block_size=block_size,
            int_num_layers_list=int_num_layers_list,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            forward_expansion=forward_expansion,

        )  # 搭建模型
        self.predictor_1 = nn.Linear(
            n_embd, vocab_size)  # FC(128, 2)

        # 用于预测目的地
        self.predictor_Des = MLP(n_embd, goal_num * 2, (512, 512, 512))
        self.rand_token = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty(1, 12, n_embd)))
        # self.token_encoder = nn.Linear(config.n_embd, config.n_embd)  # 对可学习编码进行编码
        self.token_encoder = MLP(n_embd, n_embd, (512, 512, 512))
        self.des_loss_encoder = nn.Linear(n_embd, n_embd)

        # 社交关系
        # self.nei_embedding = nn.Linear(config.vocab_size * self.past_len, config.n_embd)
        self.nei_embedding = nn.Linear(vocab_size, n_embd)
        self.social_decoder = Encoder(
            n_embd, int_num_layers_list[1], n_head, forward_expansion, islinear=False
        )

        # 阶段三新加的层
        # self.traj_decoder = nn.Linear(
        #     config.n_embd, config.vocab_size)  # 得到第十帧到第十九帧的预测轨迹
        self.traj_decoder = MLP(
            n_embd, vocab_size, (512, 512, 512)
        )  # 得到第十帧到第十九帧的预测轨迹
        self.traj_decoder_20 = MLP(
            n_embd, vocab_size, (512, 512, 512)
        )  # trajectory decoder for the 20th trajectory point  第二十帧的预测轨迹

        # 加载图片
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if dataset_name == "sdd_img":
            self.image_model, preprocess = clip.load("ViT-L/14")
            for param in self.image_model.parameters():
                param.requires_grad = False
        elif dataset_name == "sdd" or dataset_name == "sdd_world":
            self.image_model = None
        else:
            model, preprocess = clip.load("ViT-L/14")
            image = self.transform(Image.open(f"scene/{dataset_name}.jpg")).unsqueeze(0).cuda()
            self.img_feat = model.encode_image(image).to(dtype=torch.float32).detach()

        self.img_encoder = nn.Linear(768, n_embd)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # He initialization for weights
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    # 偏置初始化为0
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, nn.LayerNorm):
                # LayerNorm层保持标准初始化
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_pe(self, input, if_social=False):
        device = input.device
        if if_social:
            b, n, t, d = input.size()
        else:
            b, t, d = input.size()
        # assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(1, t + 1, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)   (1, 19)
        # if t == 9:
        if not if_social:
            pos[:, -1] = 20
        tok_emb = input  # (513, 19, 128)
        pos_emb = self.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)  (1, 19, 128)
        if if_social:
            pos_emb = pos_emb.repeat(tok_emb.shape[1], 1, 1).unsqueeze(0)

        return tok_emb + pos_emb

    def spatial_interaction(self, ped, neis, mask):
        # ped (512, 1, 64)
        # neis (512, 18, 8, 2)  N is the max number of agents of current scene
        # mask (512, 2, 2) is used to stop the attention from invalid agents

        # 对邻居进行编码
        nei_embedding = self.nei_embedding(neis)  # (512, 8, 18, 128)
        nei_embedding[:, 0, :] = ped  # (512, 18, 128)
        # nei_embedding = self.get_pe(nei_embedding, if_social=True)
        temporal_enc = self.build_pos_enc(neis.shape[2]).to(device=neis.device)  # (1, 1, 8, 128)
        nei_embedding = nei_embedding + temporal_enc  # (512, 8, 8, 128)
        _, n, t, _ = neis.shape
        nei_embeddings = nei_embedding.reshape(neis.shape[0], -1,
                            self.n_embd)  # (512, 162, 128)

        mask = mask[:, 0:1].reshape(neis.shape[0], -1, 1).repeat(1, 1, neis.shape[2]).view(neis.shape[0], -1).unsqueeze(1).repeat(1, nei_embeddings.shape[1], 1)  # (512, 1, 9) -> (512, 9, 1) -> (512, 9, 18) -> (512, 162) -> (512, 162, 1) -> (512, 162, 162)
        mask_traj = torch.tril(torch.ones((1, ped.shape[1], ped.shape[1]))).repeat(neis.shape[0], neis.shape[1], neis.shape[1]).to(device=neis.device)  # (512, 162, 162)
        mask = mask * mask_traj  # (512, 162, 162)
        int_feat = self.social_decoder(
            nei_embeddings, nei_embeddings, mask
        )  # [B K embed_size]  (512, 112, 128)

        # return int_feat.reshape(neis.shape[0], n, t, -1)[:, 0]
        return int_feat.contiguous().view(neis.shape[0], n, t, -1)[:, 0]

    def loss_function(self, pred, gt):  # pred:[B, 20, 2]
        # joint loss
        JL = self.joint_loss(
            pred) if self.lambda_j > 0 else 0.0  # 多样性损失
        RECON = self.recon_loss(
            pred, gt) if self.lambda_recon > 0 else 0.0  # 目的地损失
        # print('JL', JL * self.lambda_j, 'RECON', RECON * self.lambda_recon)
        loss = JL * self.lambda_j + RECON * self.lambda_recon  # 总损失
        return loss

    def joint_loss(self, pred):  # pred:[B, 20, 2]  多样性损失
        loss = 0.0
        for Y in pred:  # 取出来每个行人预测的20个目的地
            # 计算每两个目的地之间的距离  20个目的地两两组合,共有(19 * 20) / 2 = 190个组合
            dist = F.pdist(Y, 2) ** 2
            # 计算每个组合的损失,并取平均
            loss += (-dist / 1).exp().mean()
        loss /= pred.shape[0]  # 取所有行人预测的损失的平均
        return loss

    def recon_loss(self, pred, gt):
        distances = torch.norm(pred - gt, dim=2)
        # 找到每个行人预测的20个目的地中,与真实目的地距离最小的那个
        index_min = torch.argmin(distances, dim=1)
        # 计算每个行人预测的20个目的地中,与真实目的地距离最小的那个距离
        min_distances = distances[torch.arange(0, len(index_min)), index_min]
        # 计算所有行人预测的20个目的地中,与真实目的地距离最小的那个距离的平均
        loss_recon = torch.sum(min_distances) / distances.shape[0]
        return loss_recon

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.n_embd)  # (8, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (8, 1)
        div_term = torch.exp(torch.arange(0, self.n_embd, 2).float() * (-np.log(10000.0) / self.n_embd))  # (64,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 128)

    def get_trajectory(self, traj_norm, neis, mask, scene):
        predictions = torch.Tensor().cuda()

        # 对输入轨迹进行编码
        past_state = self.traj_encoder(traj_norm)

        # 融合场景信息
        if "sdd" not in self.dataset_name:
            img_feat = self.img_encoder(self.img_feat).unsqueeze(1)
            final_img_feat = repeat(img_feat, "() n d -> b n d", b=neis.shape[0])
        elif self.dataset_name == "sdd" or self.dataset_name == "sdd_world":
            final_img_feat = torch.zeros(neis.shape[0], 1, self.n_embd).to(device=traj_norm.device)
        else:
            scene = scene.to(device=traj_norm.device)
            img_feat = self.image_model.encode_image(scene).to(dtype=torch.float32).detach()
            img_feat = self.img_encoder(img_feat).unsqueeze(1)
            final_img_feat = img_feat

        # 提取社会交互信息
        int_feat = self.spatial_interaction(
            past_state[:, :self.past_len], neis[:, :, :self.past_len], mask)  # (512, 8, 128)
        int_feat = int_feat + final_img_feat
        past_state[:, :self.past_len] = int_feat  # (512, 20, 128)

        # 本次使用的真实值
        past_feat = int_feat

        # 先预测目的地
        des_token = repeat(
            self.rand_token[:, -1:], "() n d -> b n d", b=past_feat.size(0)
        )  # (513, 1, 128)  可学习编码
        des_state = self.token_encoder(des_token)  # (513, 1, 128)  对可学习编码进行编码

        des_input = torch.cat((past_feat, des_state), dim=1)
        des_input = self.get_pe(des_input)
        des_feat = self.AR_Model(
            des_input
        )  # (514, 9, 128)
        des_feat = des_feat + final_img_feat
        pred_des = self.predictor_Des(
            des_feat[:, -1]
        )  # generate 20 destinations for each trajectory  (512, 1, 40)  每条轨迹生成20个目的地
        pred_des = pred_des.view(pred_des.size(0), self.goal_num, -1)

        # generate N=20 future trajectories
        for i in range(self.goal_num):
            fut_token = repeat(
                self.rand_token[:, :-1], "() n d -> b n d", b=traj_norm.size(0)
            )

            fut_feat = self.token_encoder(fut_token)
            des_feat = self.traj_encoder(pred_des[:, i])
            traj_feat = torch.cat(
                (past_feat, fut_feat, des_feat.unsqueeze(1)), 1)  # (512, 20, 128)

            traj_feat = self.get_pe(traj_feat)
            prediction_feat = self.AR_Model(traj_feat, mask_type="all")  # (512, 20, 128)
            prediction_feat = prediction_feat + final_img_feat

            mid_prediction = self.traj_decoder(
                prediction_feat[:, self.past_len:-1])  # (512, 11, 2)
            des_prediction = self.traj_decoder_20(
                prediction_feat[:, -1]
            ) + pred_des[:, i]
            total_prediction = torch.cat(
                (mid_prediction, des_prediction.unsqueeze(1)), 1
            )

            prediction_single = total_prediction
            predictions = torch.cat(
                (predictions, prediction_single.unsqueeze(1)), dim=1
            )
        return predictions

    def forward(self, traj_norm, neis, mask, destination, scene):
        loss = torch.tensor(0.0, device=traj_norm.device)
        # 对输入轨迹进行编码
        past_state = self.traj_encoder(traj_norm)  # (513, 20, 128)

        # 融合场景信息
        if "sdd" not in self.dataset_name:
            img_feat = self.img_encoder(self.img_feat).unsqueeze(1)
            final_img_feat = repeat(img_feat, "() n d -> b n d", b=neis.shape[0])
        elif self.dataset_name == "sdd" or self.dataset_name == "sdd_world":
            final_img_feat = torch.zeros(neis.shape[0], 1, self.n_embd).to(device=traj_norm.device)
        else:
            scene = scene.to(device=traj_norm.device)
            img_feat = self.image_model.encode_image(scene).to(dtype=torch.float32)
            img_feat = self.img_encoder(img_feat).unsqueeze(1).detach()
            final_img_feat = img_feat

        # 提取社会交互信息
        int_feat = self.spatial_interaction(past_state[:, :self.past_len], neis[:, :, :self.past_len], mask)  # (512, 8, 128)
        int_feat = int_feat + final_img_feat
        # past_state[:, :self.past_len] = int_feat  # (512, 20, 128)

        # 本次使用的真实值
        # past_feat = past_state[:, :self.total_len - c_pred_len - 1]
        past_feat = int_feat

        # 先预测目的地
        des_token = repeat(
            self.rand_token[:, -1:], "() n d -> b n d", b=past_feat.size(0)
        )  # (513, 1, 128)  可学习编码
        des_state = self.token_encoder(des_token)  # (513, 1, 128)  对可学习编码进行编码

        # 位置编码
        des_input = torch.cat((past_feat, des_state), dim=1)
        des_feat = self.get_pe(des_input)
        des_feat = self.AR_Model(
            des_feat, mask_type="causal"
        )  # (514, 9, 128)
        des_feat = des_feat + final_img_feat
        pred_des = self.predictor_Des(
            des_feat[:, -1]
        )  # generate 20 destinations for each trajectory  (512, 1, 40)  每条轨迹生成20个目的地
        pred_des = pred_des.view(pred_des.size(0), self.goal_num, -1)  # (512, 20, 2)

        # 目的地损失
        true_des_feat = self.traj_encoder(destination.squeeze())  # (514, 128) 对预测的目的地进行编码
        pred_des_feat = self.des_loss_encoder(des_feat[:, -1])  # (514, 128) 对预测的目的地进行编码

        loss += F.mse_loss(pred_des_feat, true_des_feat) * self.lambda_des
        loss += self.loss_function(pred_des, destination) * self.lambda_des

        # 从20个预测目的地中找到和真实目的地最接近的目的地
        distances = torch.norm(destination - pred_des, dim=2)  # (514, 20)  计算每个预测目的地与真实目的地的距离
        index_min = torch.argmin(distances, dim=1)  # (514)  找到每个轨迹的最小距离的索引
        min_des_traj = pred_des[torch.arange(0, len(index_min)), index_min]  # (514, 2)  找到每个轨迹的最小距离的目的地
        destination_prediction = min_des_traj  # (514, 2)  预测的目的地

        # 本次预测的观察帧
        traj_input = past_feat[:, :self.past_len]

        for c_pred_len in range(1, self.future_len):
            # 本次预测的帧id
            pred_frame_id = [v for v in range((self.past_len),(self.past_len+c_pred_len))]

            # 预测帧token
            fut_token = self.rand_token[0, [v-8 for v in pred_frame_id]].unsqueeze(0)
            fut_token = repeat(
            fut_token, "() n d -> b n d", b=traj_input.size(0)
            )  # (514, 11, 128)  可学习编码
            fut_feat = self.token_encoder(fut_token)

            # 目的地  训练用真实目的地
            des = self.traj_encoder(destination.squeeze())  # (514, 128) 对预测的目的地进行编码

            # 拼接 观察帧轨迹 + 可学习编码 + 预测的目的地编码
            concat_traj_feat = torch.cat((traj_input, fut_feat, des.unsqueeze(1)), 1)  # (514, 10, 128)
            concat_traj_feat = self.get_pe(concat_traj_feat)
            prediction_feat = self.AR_Model(concat_traj_feat, mask_type="all")  # (514, 10, 128)  Transformer  没有用mask
            prediction_feat = prediction_feat + final_img_feat
            pred_traj = self.traj_decoder(prediction_feat[:, self.past_len:-1])  # (514, 2)  预测的中间轨迹

            # 对第19帧进行编码  得到第二十帧的预测轨迹
            des_prediction = self.traj_decoder_20(
                prediction_feat[:, -1]
            ) + destination_prediction  # (514, 1, 2)  预测终点的残差

            # 拼接预测轨迹
            pred_results = torch.cat(
                (pred_traj, des_prediction.unsqueeze(1)), 1
            )

            # 计算轨迹损失
            traj_gt = traj_norm[:, pred_frame_id[0]:pred_frame_id[-1]+1]  # 中间轨迹
            traj_gt = torch.cat((traj_gt, traj_norm[:, -1].unsqueeze(1)), 1)  # 加上终点

            loss += F.mse_loss(pred_results, traj_gt)

        return loss