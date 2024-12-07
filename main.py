import os, sys, time
import importlib
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from data import Dataloader
from utils import seed, get_rng_state, set_rng_state
from models.model import Final_Model


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", default=[])
    parser.add_argument("--test", nargs="+", default=[])
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=717)
    parser.add_argument("--no-fpc", action="store_true", default=False)
    parser.add_argument("--fpc-finetune", action="store_true", default=False)

    return parser.parse_args()


def test(model: Final_Model, fpc=1):
    sys.stdout.write("\r\033[K Evaluating...{}/{}".format(0, len(test_dataset)))
    tic = time.time()
    model.eval()
    ADE, FDE = [], []
    set_rng_state(init_rng_state, settings.device)
    batch = 0
    fpc = int(fpc) if fpc else 1
    fpc_config = "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC"
    y_true, y_pred = [], []
    with torch.no_grad():
        for _, (ped, neis, mask, initial_pos, scene) in  enumerate(test_data):
            ped, neis, mask, initial_pos = (
                ped.to(settings.device),
                neis.to(settings.device),
                mask.to(settings.device),
                initial_pos.to(settings.device),
            )  # (512, 20, 2)  (512, 1, 20, 2)  (512, 1, 1)  (512, 1, 2)

            if config.DATASET_NAME == "eth":
                ped[:, :, 0] = ped[:, :, 0] * config.DATA_SCALING[0]
                ped[:, :, 1] = ped[:, :, 1] * config.DATA_SCALING[1]

            scale = torch.randn(ped.shape[0]) * 0.05 + 1
            scale = scale.to(settings.device)
            scale = scale.reshape(ped.shape[0], 1, 1)  # (512, 1, 1)
            ped = ped * scale
            scale = scale.reshape(ped.shape[0], 1, 1, 1)
            neis = neis * scale

            traj_norm = ped  # 减去第八帧做归一化  (513, 20, 2)
            output = model.get_trajectory(traj_norm, neis, mask, scene)
            output = output.data

            future_rep = traj_norm[:, 8:-1, :].unsqueeze(1).repeat(1, model.goal_num, 1, 1)
            future_goal = traj_norm[:, -1:, :].unsqueeze(1).repeat(1, model.goal_num, 1, 1)
            future = torch.cat((future_rep, future_goal), dim=2)
            distances = torch.norm(output - future, dim=3)

            fde_mean_distances = torch.mean(distances[:, :, -1:], dim=2) # find the tarjectory according to the last frame's distance
            fde_index_min = torch.argmin(fde_mean_distances, dim=1)
            fde_min_distances = distances[torch.arange(0, len(fde_index_min)), fde_index_min]
            FDE.append(fde_min_distances[:, -1])

            ade_mean_distances = torch.mean(distances[:, :, :], dim=2) # find the tarjectory according to the last frame's distance
            ade_index_min = torch.argmin(ade_mean_distances, dim=1)
            ade_min_distances = distances[torch.arange(0, len(ade_index_min)), ade_index_min]
            ADE.append(torch.mean(ade_min_distances, dim=1))

    ADE = torch.cat(ADE)
    FDE = torch.cat(FDE)
    if torch.is_tensor(config.WORLD_SCALE) or config.WORLD_SCALE != 1:
        if not torch.is_tensor(config.WORLD_SCALE):
            config.WORLD_SCALE = torch.as_tensor(
                config.WORLD_SCALE, device=ADE.device, dtype=ADE.dtype
            )
        ADE *= config.WORLD_SCALE
        FDE *= config.WORLD_SCALE
    ade = ADE.mean()
    fde = FDE.mean()
    sys.stdout.write(
        "\r\033[K ADE: {:.4f}; FDE: {:.4f} ({}) -- time: {}s".format(
            ade, fde, fpc_config, int(time.time() - tic)
        )
    )
    print()
    return ade, fde

if __name__ == "__main__":
    settings = parse_args()
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.device = torch.device(settings.device)

    if torch.cuda.is_available():  # 设置GPU编号
        torch.cuda.set_device(settings.gpu)

    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
        batch_first=False,
        frameskip=settings.frameskip,
        ob_horizon=config.OB_HORIZON,
        pred_horizon=config.PRED_HORIZON,
        device=settings.device,
        seed=settings.seed,
    )
    train_data, test_data = None, None

    # 测试数据
    if settings.test:
        print(settings.test)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
        test_dataset = Dataloader(
            settings.test,
            **kwargs,
            inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE,
            shuffle=False,

        )
        test_data = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=test_dataset.coll_fn,
            batch_sampler=test_dataset.batch_sampler,
        )

    # 训练数据
    if settings.train:
        print(settings.train)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train))]
        else:
            inclusive = None
        train_dataset = Dataloader(
            settings.train,
            **kwargs,
            inclusive_groups=inclusive,
            rotation=config.ROTATION,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            # batches_per_epoch=config.EPOCH_BATCHES
        )

        train_data = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=train_dataset.coll_fn,
            batch_sampler=train_dataset.batch_sampler,
        )
        batches = train_dataset.batches_per_epoch

    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    model = Final_Model(
        dataset_name=config.DATASET_NAME,
        past_len=config.OB_HORIZON,
        future_len=config.PRED_HORIZON,
        n_embd=config.N_EMBD,
        dropout=config.DROPOUT,
        int_num_layers_list=config.INT_NUM_LAYERS_LIST,
        n_head=config.N_HEAD,
        forward_expansion=config.FORWARD_EXPANSION,
        block_size=config.BLOCK_SIZE,
        vocab_size=config.VOCAB_SIZE,
        goal_num=config.GOAL_NUM,
        lambda_des=config.LAMBDA_DES,
    )
    model.to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.LEARNING_RATE_MIN
        )
    start_epoch = 0
    if settings.ckpt:
        ckpt = os.path.join(settings.ckpt, "ckpt-last")
        ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            ade_best = state_dict["ade"]
            fde_best = state_dict["fde"]
            fpc_best = state_dict["fpc"] if "fpc" in state_dict else 1
        else:
            ade_best = 100000
            fde_best = 100000
            fpc_best = 1
        if train_data is None:  # testing mode
            ckpt = ckpt_best
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=settings.device)
            model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
                rng_state = [
                    r.to("cpu") if torch.is_tensor(r) else r
                    for r in state_dict["rng_state"]
                ]
            start_epoch = state_dict["epoch"]
    end_epoch = (
        start_epoch + 1
        if train_data is None or start_epoch >= config.EPOCHS
        else config.EPOCHS
    )

    if settings.train and settings.ckpt:
        logger = SummaryWriter(log_dir=settings.ckpt)
    else:
        logger = None

    if train_data is not None:
        log_str = (
            "\r\033[K {cur_batch:>"
            + str(len(str(batches)))
            + "}/"
            + str(batches)
            + " [{done}{remain}] -- time: {time}s - {comment}"
        )
        progress = 20 / batches if batches > 20 else 1
        optimizer.zero_grad()

    for epoch in range(start_epoch + 1, end_epoch + 1):
        ###############################################################################
        #####                                                                    ######
        ##### train                                                              ######
        #####                                                                    ######
        ###############################################################################
        losses = None
        if train_data is not None and epoch <= config.EPOCHS:
            print("Epoch {}/{}".format(epoch, config.EPOCHS))
            tic = time.time()
            set_rng_state(rng_state, settings.device)
            losses = {}
            model.train()
            sys.stdout.write(
                log_str.format(
                    cur_batch=0,
                    done="",
                    remain="." * int(batches * progress),
                    time=round(time.time() - tic),
                    comment="",
                )
            )
            train_loss = 0

            # 训练模型
            for batch, (ped, neis, mask, initial_pos, scene) in  enumerate(train_data):
                ped, neis, mask, initial_pos = (
                    ped.to(settings.device),
                    neis.to(settings.device),
                    mask.to(settings.device),
                    initial_pos.to(settings.device),
                )  # (512, 20, 2)  (512, 1, 20, 2)  (512, 1, 1)  (512, 1, 2)

                if config.DATASET_NAME == "eth":
                    ped[:, :, 0] = ped[:, :, 0] * config.DATA_SCALING[0]
                    ped[:, :, 1] = ped[:, :, 1] * config.DATA_SCALING[1]

                scale = torch.randn(ped.shape[0]) * 0.05 + 1
                scale = scale.to(settings.device)
                scale = scale.reshape(ped.shape[0], 1, 1)  # (512, 1, 1)
                ped = ped * scale
                scale = scale.reshape(ped.shape[0], 1, 1, 1)
                neis = neis * scale

                nei_obs = neis[:, :, :config.OB_HORIZON]  # (512, 2, 8, 2)  邻居的观察帧

                traj_norm = ped  # 减去第八帧做归一化  (513, 20, 2)
                x = traj_norm[:, : config.OB_HORIZON, :]  # 前8帧数据 (513, 8, 2)  观察帧
                destination = traj_norm[:, -1:, :]  # 最后一帧数据 (513, 1, 2)  目的地
                y = traj_norm[:, config.OB_HORIZON :, :]  # 后12帧数据 (513, 12, 2)  预测帧

                trajectory = traj_norm + initial_pos  # 加上初始位置  (512, 20, 2)
                abs_past = trajectory[:, : config.OB_HORIZON, :]  # 前8帧数据 (512, 8, 2)  未归一化版本
                initial_pose = trajectory[:, config.OB_HORIZON - 1, :]  # 第八帧数据 (512, 2)  未归一化

                loss = torch.tensor(0.0, device=settings.device)
                loss = model(traj_norm, neis, mask, destination, scene)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                sys.stdout.write(
                    log_str.format(
                        cur_batch=batch + 1,
                        done="=" * int((batch + 1) * progress),
                        remain="."
                        * (int(batches * progress) - int((batch + 1) * progress)),
                        time=round(time.time() - tic),
                        comment=" - ".join(
                            ["{}: {:.4f}".format(k, v) for k, v in losses.items()]
                        ),
                    )
                )
            rng_state = get_rng_state(settings.device)
            print()

        scheduler.step()
        ###############################################################################
        #####                                                                    ######
        ##### test                                                               ######
        #####                                                                    ######
        ###############################################################################
        ade, fde = 10000, 10000
        perform_test = (
            train_data is None or epoch >= config.TEST_SINCE
        ) and test_data is not None
        if perform_test:
            if (
                not settings.no_fpc
                and not settings.fpc_finetune
                and losses is None
                and fpc_best > 1
            ):
                fpc = fpc_best
            else:
                fpc = 1
            ade, fde = test(model, fpc)

        ###############################################################################
        #####                                                                    ######
        ##### log                                                                ######
        #####                                                                    ######
        ###############################################################################
        if losses is not None and settings.ckpt:
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar("train/{}".format(k), v, epoch)
                if perform_test:
                    logger.add_scalar("eval/ADE", ade, epoch)
                    logger.add_scalar("eval/FDE", fde, epoch)
            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ade=ade,
                fde=fde,
                epoch=epoch,
                rng_state=rng_state,
            )
            torch.save(state, ckpt)
            if ade < ade_best:
                ade_best = ade
                fde_best = fde
                state = dict(model=state["model"], ade=ade, fde=fde, epoch=epoch)
                torch.save(state, ckpt_best)

            sys.stdout.write(
                "\r\033[K Best ADE: {:.4f}; Best FDE: {:.4f}".format(ade_best, fde_best)
            )
            print()
