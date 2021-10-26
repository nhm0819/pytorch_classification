from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import torch
import cv2
import time


class WireDataset(Dataset):
    def __init__(self, args, df, transforms=None):
        super().__init__()
        self.args = args
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.num_classes = args.num_classes


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):


        # # parquet file을 교체하는 refresh_idx를 전체 주기보다 짧게 디자인한다.
        # # 이때 교체 주기는 하이퍼파라미터로 중복 비율에 영향을 미친다.
        # parquet_refresh_freq = 4
        # refresh_idx = 1 \
        #     if self.args.num_workers == 0 \
        #     else len(self.current_pd_parquets) \
        #     - len(self.current_pd_parquets) // (parquet_refresh_freq*self.num_workers)
        #
        # if len(self.current_indices_in_cache) < refresh_idx:
        #     self.current_parquet_idx += 1
        #     # 캐시 파일 교체 주기가 빨라지므로 교체할 때 parquet list도 무작위로 섞는다.
        #     if self.current_parquet_idx >= self.steps_cache:
        #         self.current_parquet_idx = 0
        #         if self.num_workers > 0:
        #             now = time.time()
        #             seed = int((now - int(now))*100000)
        #             rng = np.random.RandomState(seed=seed)
        #             rng.shuffle(self.parquet_list)
        #
        #     self._cache_setting()
        #
        # # 단일 프로세스에서 같은 리스트를 pop으로 꺼내는 경우는 병렬 프로세스에서 중복을 일으키므로
        # # indices 행렬에서 random sampling으로 수행 후 해당 index를 삭제하는 형태로 변형한다.
        # if self.num_workers != 0:
        #     now = time.time()
        #     seed = int((now - int(now))*100000)
        #     rng = np.random.RandomState(seed=seed)
        #     rand_idx = rng.randint(
        #         len(self.current_indices_in_cache)
        #         )
        #     pd_idx = self.current_indices_in_cache[rand_idx]
        #     del self.current_indices_in_cache[rand_idx]
        # else:
        #     pd_idx = self.current_indices_in_cache.pop()
        # pd_raw = self.current_pd_parquets.iloc[pd_idx]
        #
        # sample = self._transform_raw_to_array(pd_raw)
        #
        # if self.transform:
        #     transformed = self.transform(image=sample['img_input'])
        #     sample['img_input'] = transformed['image']


        img_path = os.path.join(self.args.dataset_dir, self.df["path"][idx])
        img_path = img_path.replace("/", "\\")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']

        label = self.df["label"][idx]

        return img, label



class TestDataset(Dataset):
    def __init__(self, args, df, transforms=None):
        super().__init__()
        self.args = args
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.num_classes = args.num_classes


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.args.dataset_dir, self.df["path"][idx])
        img_path = img_path.replace("/", "\\")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']

        label = self.df["label"][idx]

        return img, label, img_path

