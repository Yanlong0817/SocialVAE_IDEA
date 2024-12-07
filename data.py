from typing import Optional, Sequence, List

import os, sys
import torch
import numpy as np

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import random


class Dataloader(torch.utils.data.Dataset):

    class FixedNumberBatchSampler(torch.utils.data.sampler.BatchSampler):
        def __init__(self, n_batches, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_batches = n_batches
            self.sampler_iter = None  # iter(self.sampler)

        def __iter__(self):
            # same with BatchSampler, but StopIteration every n batches
            counter = 0
            batch = []
            while True:
                if counter >= self.n_batches:
                    break
                if self.sampler_iter is None:
                    self.sampler_iter = iter(self.sampler)
                try:
                    idx = next(self.sampler_iter)
                except StopIteration:
                    self.sampler_iter = None
                    if self.drop_last:
                        batch = []
                    continue
                batch.append(idx)
                if len(batch) == self.batch_size:
                    counter += 1
                    yield batch
                    batch = []

    def __init__(
        self,
        files: List[str],
        ob_horizon: int,
        pred_horizon: int,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        batches_per_epoch=None,
        frameskip: int = 1,
        inclusive_groups: Optional[Sequence] = None,
        batch_first: bool = False,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        rotation: bool = False,
        dist_threshold: int = 2,
        use_augmentation: bool = False,
    ):
        super().__init__()
        self.ob_horizon = ob_horizon
        self.pred_horizon = pred_horizon
        self.horizon = self.ob_horizon + self.pred_horizon
        self.dist_threshold = dist_threshold
        self.frameskip = int(frameskip) if frameskip and int(frameskip) > 1 else 1
        self.batch_first = batch_first
        self.rotation = rotation
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        else:
            self.device = device

        if inclusive_groups is None:
            inclusive_groups = [[] for _ in range(len(files))]
        assert len(inclusive_groups) == len(files)

        print(" Scanning files...")
        files_ = []
        for path, incl_g in zip(files, inclusive_groups):
            if os.path.isdir(path):
                files_.extend(
                    [
                        (os.path.join(root, f), incl_g)
                        for root, _, fs in os.walk(path)
                        for f in fs
                        if f.endswith(".txt")
                    ]
                )
            elif os.path.exists(path):
                files_.append((path, incl_g))
        data_files = sorted(files_, key=lambda _: _[0])

        data = []

        done = 0
        # too large of max_workers will cause the problem of memory usage
        max_workers = min(len(data_files), torch.get_num_threads(), 20)
        with ProcessPoolExecutor(
            mp_context=multiprocessing.get_context("spawn"), max_workers=max_workers
        ) as p:
            futures = [
                p.submit(self.__class__.load, self, f, incl_g)
                for f, incl_g in data_files
            ]
            for fut in as_completed(futures):
                done += 1
                sys.stdout.write(
                    "\r\033[K Loading data files...{}/{}".format(done, len(data_files))
                )
            for fut in futures:
                item = fut.result()
                if item is not None:
                    data.extend(item)
                sys.stdout.write(
                    "\r\033[K Loading data files...{}/{} ".format(done, len(data_files))
                )
        # 模仿PPT扩充数据集
        raw_len = len(data)
        if use_augmentation:
            for i in range(raw_len):
                traj = data[i]

                # 旋转
                ks = [1, 2, 3]
                for k in ks:
                    traj_rot = []
                    traj_rot.append(self.rot(traj[0][:, :2], k))
                    traj_rot.append(self.rot(traj[1], k))
                    traj_rot.append(self.rot(traj[2][:, :, :2], k))
                    data.append(tuple(traj_rot))

                # 水平翻转
                traj_flip = []
                traj_flip.append(self.fliplr(traj[0][:, :2]))
                traj_flip.append(self.fliplr(traj[1]))
                traj_flip.append(self.fliplr(traj[2][:, :, :2]))
                data.append(tuple(traj_flip))
        self.data = np.array(data, dtype=object)
        del data
        print("\n   {} trajectories loaded.".format(raw_len))

        self.rng = np.random.RandomState()
        if seed:
            self.rng.seed(seed)

        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(self)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(self)
        if batches_per_epoch is None:
            self.batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, batch_size, drop_last
            )
            self.batches_per_epoch = len(self.batch_sampler)
        else:
            self.batch_sampler = self.__class__.FixedNumberBatchSampler(
                batches_per_epoch, sampler, batch_size, drop_last
            )
            self.batches_per_epoch = batches_per_epoch

    @staticmethod
    def rot(data, k=1):
        """
        Rotates image and coordinates counter-clockwise by k * 90° within image origin
        :param df: Pandas DataFrame with at least columns 'x' and 'y'
        :param image: PIL Image
        :param k: Number of times to rotate by 90°
        :return: Rotated Dataframe and image
        """
        data_ = data.copy()

        c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
        R = np.array([[c, s], [-s, c]])  # 旋转矩阵
        data_ = np.dot(data_, R)  # 旋转数据
        return data_

    @staticmethod
    def fliplr(data):
        """
        Flip image and coordinates horizontally
        :param df: Pandas DataFrame with at least columns 'x' and 'y'
        :param image: PIL Image
        :return: Flipped Dataframe and image
        """
        data_ = data.copy()
        R = np.array([[-1, 0], [0, 1]])
        data_ = np.dot(data_, R)

        return data_

    def coll_fn(self, scenario_list):
        # batch <list> [[ped, neis]]]
        ped, neis = [], []
        shift = []

        n_neighbors = []

        for item in scenario_list:
            ped_obs_traj, ped_pred_traj, neis_traj = (
                item[0],
                item[1],
                item[2],
            )  # [T 2] [N T 2] N is not a fixed number  取出来观察帧,预测帧,邻居轨迹

            # 拼接轨迹
            ped_traj = np.concatenate(
                (ped_obs_traj[:, :2], ped_pred_traj), axis=0
            )  # (20, 2) 拼接完整的行人轨迹
            neis_traj = neis_traj[:, :, :2].transpose(
                1, 0, 2
            )  # (N, 20, 2) 邻居轨迹  N表示邻居数量 可能为0
            neis_traj = np.concatenate(
                (np.expand_dims(ped_traj, axis=0), neis_traj), axis=0
            )  # (1+N, 20, 2)  行人和邻居轨迹

            # 计算行人和邻居之间的距离
            distance = np.linalg.norm(
                np.expand_dims(ped_traj, axis=0) - neis_traj, axis=-1
            )  # (1+N, 20)  计算行人和邻居之间的距离
            distance = distance[:, : self.ob_horizon]  # 取出来前八帧的距离  (1+N, 8)
            distance = np.mean(distance, axis=-1)  # mean distance  取出来和每个邻居的观察帧的平均距离  (1+N, )
            # distance = distance[:, -1] # final distance
            neis_traj = neis_traj[distance < self.dist_threshold]  # 取出来距离小于阈值的邻居轨迹

            n_neighbors.append(neis_traj.shape[0])  # 邻居数目,若只有1个邻居,则表示该行人本身

            origin = ped_traj[self.ob_horizon - 1 : self.ob_horizon]  # 取出来行人的第八帧观察帧数据  (1, 2)
            ped_traj = ped_traj - origin  # 当前行人减去观察帧数据
            if neis_traj.shape[0] != 0:
                neis_traj = neis_traj - np.expand_dims(origin, axis=0)  # 邻居减去当前行人观察帧数据

            shift.append(origin)

            if self.rotation:  # 旋转数据
                angle = random.random() * np.pi  # 随机旋转的角度
                rot_mat = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                ped_traj = np.matmul(ped_traj, rot_mat)  # 旋转行人轨迹
                if neis_traj.shape[0] != 0:
                    rot_mat = np.expand_dims(rot_mat, axis=0)
                    rot_mat = np.repeat(rot_mat, neis_traj.shape[0], axis=0)
                    neis_traj = np.matmul(neis_traj, rot_mat)  # 旋转邻居轨迹

            ped.append(ped_traj)
            neis.append(neis_traj)

        max_neighbors = max(n_neighbors)  # 当前batch最大邻居数目
        neis_pad = []
        neis_mask = []
        for neighbor, n in zip(neis, n_neighbors):  # 遍历每个行人的邻居
            neis_pad.append(
                np.pad(neighbor, ((0, max_neighbors - n), (0, 0), (0, 0)), "constant")
            )  # 邻居轨迹填充成相同的长度
            mask = np.zeros((max_neighbors, max_neighbors))
            mask[:n, :n] = 1  # mask表示是否有邻居, 若为0表示没有邻居, 是填充值
            neis_mask.append(mask)

        ped = np.stack(ped, axis=0)  # (512, 20, 2)  512表示batch_size
        neis = np.stack(
            neis_pad, axis=0
        )  # (512, 1, 20, 2)  邻居的轨迹  1表示当前batch的行人最多只有一个邻居
        neis_mask = np.stack(neis_mask, axis=0)  # (512, 1, 1)  mask表示是否有邻居
        shift = np.stack(shift, axis=0)  # (512, 1, 2)  第八帧数据

        ped = torch.tensor(ped, dtype=torch.float32)
        neis = torch.tensor(neis, dtype=torch.float32)
        neis_mask = torch.tensor(neis_mask, dtype=torch.int32)
        shift = torch.tensor(shift, dtype=torch.float32)  # 第八帧数据
        return ped, neis, neis_mask, shift, None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load(self, filename, inclusive_groups):
        if os.path.isdir(filename):
            return None

        horizon = (self.horizon - 1) * self.frameskip
        with open(filename, "r") as record:
            data = self.load_traj(record)
        data = self.extend(data, self.frameskip)
        time = np.sort(list(data.keys()))
        if len(time) < horizon + 1:
            return None
        valid_horizon = self.ob_horizon + self.pred_horizon

        traj = []
        e = len(time)
        tid0 = 0
        while tid0 < e - horizon:
            tid1 = tid0 + horizon
            t0 = time[tid0]

            idx = [
                aid
                for aid, d in data[t0].items()
                if not inclusive_groups or any(g in inclusive_groups for g in d[-1])
            ]
            if idx:
                idx_all = list(data[t0].keys())
                for tid in range(tid0 + self.frameskip, tid1 + 1, self.frameskip):
                    t = time[tid]
                    idx_cur = [
                        aid
                        for aid, d in data[t].items()
                        if not inclusive_groups
                        or any(g in inclusive_groups for g in d[-1])
                    ]
                    if not idx_cur:  # ignore empty frames
                        tid0 = tid
                        idx = []
                        break
                    idx = np.intersect1d(idx, idx_cur)
                    if len(idx) == 0:
                        break
                    idx_all.extend(data[t].keys())
            if len(idx):
                data_dim = 6
                neighbor_idx = np.setdiff1d(idx_all, idx)
                if len(idx) == 1 and len(neighbor_idx) == 0:
                    agents = np.array(
                        [
                            [data[time[tid]][idx[0]][:data_dim]] + [[1e9] * data_dim]
                            for tid in range(tid0, tid1 + 1, self.frameskip)
                        ]
                    )  # L x 2 x 6
                else:
                    agents = np.array(
                        [
                            [data[time[tid]][i][:data_dim] for i in idx]
                            + [
                                (
                                    data[time[tid]][j][:data_dim]
                                    if j in data[time[tid]]
                                    else [1e9] * data_dim
                                )
                                for j in neighbor_idx
                            ]
                            for tid in range(tid0, tid1 + 1, self.frameskip)
                        ]
                    )  # L X N x 6
                for i in range(len(idx)):
                    hist = agents[: self.ob_horizon, i]  # L_ob x 6
                    future = agents[
                        self.ob_horizon : valid_horizon, i, :2
                    ]  # L_pred x 2
                    neighbor = agents[
                        :valid_horizon, [d for d in range(agents.shape[1]) if d != i]
                    ]  # L x (N-1) x 6
                    traj.append((hist, future, neighbor))
            tid0 += 1

        items = []
        for hist, future, neighbor in traj:
            hist = np.float32(hist)
            future = np.float32(future)
            neighbor = np.float32(neighbor)
            items.append((hist, future, neighbor))
        return items

    def extend(self, data, frameskip):
        time = np.sort(list(data.keys()))
        dts = np.unique(time[1:] - time[:-1])
        dt = dts.min()
        if np.any(dts % dt != 0):
            raise ValueError("Inconsistent frame interval:", dts)
        i = 0
        while i < len(time) - 1:
            if time[i + 1] - time[i] != dt:
                time = np.insert(time, i + 1, time[i] + dt)
            i += 1
        # ignore those only appearing at one frame
        for tid, t in enumerate(time):
            removed = []
            if t not in data:
                data[t] = {}
            for idx in data[t].keys():
                t0 = time[tid - frameskip] if tid >= frameskip else None
                t1 = time[tid + frameskip] if tid + frameskip < len(time) else None
                if (t0 is None or t0 not in data or idx not in data[t0]) and (
                    t1 is None or t1 not in data or idx not in data[t1]
                ):
                    removed.append(idx)
            for idx in removed:
                data[t].pop(idx)
        # extend v
        for tid in range(len(time) - frameskip):
            t0 = time[tid]
            t1 = time[tid + frameskip]
            if t1 not in data or t0 not in data:
                continue
            for i, item in data[t1].items():
                if i not in data[t0]:
                    continue
                x0 = data[t0][i][0]
                y0 = data[t0][i][1]
                x1 = data[t1][i][0]
                y1 = data[t1][i][1]
                vx, vy = x1 - x0, y1 - y0
                data[t1][i].insert(2, vx)
                data[t1][i].insert(3, vy)
                if tid < frameskip or i not in data[time[tid - 1]]:
                    data[t0][i].insert(2, vx)
                    data[t0][i].insert(3, vy)
        # extend a
        for tid in range(len(time) - frameskip):
            t_1 = None if tid < frameskip else time[tid - frameskip]
            t0 = time[tid]
            t1 = time[tid + frameskip]
            if t1 not in data or t0 not in data:
                continue
            for i, item in data[t1].items():
                if i not in data[t0]:
                    continue
                vx0 = data[t0][i][2]
                vy0 = data[t0][i][3]
                vx1 = data[t1][i][2]
                vy1 = data[t1][i][3]
                ax, ay = vx1 - vx0, vy1 - vy0
                data[t1][i].insert(4, ax)
                data[t1][i].insert(5, ay)
                if t_1 is None or i not in data[t_1]:
                    # first appearing frame, pick value from the next frame
                    data[t0][i].insert(4, ax)
                    data[t0][i].insert(5, ay)
        return data

    def load_traj(self, file):
        data = {}
        for row in file.readlines():
            item = row.split()
            if not item:
                continue
            t = int(float(item[0]))
            idx = int(float(item[1]))
            x = float(item[2])
            y = float(item[3])
            group = item[4].split("/") if len(item) > 4 else None
            if t not in data:
                data[t] = {}
            data[t][idx] = [x, y, group]
        return data
