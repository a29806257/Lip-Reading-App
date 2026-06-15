import json
import os
import random
from PIL import Image
from typing import List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

class VideoAugmentation:
    def __init__(self, size=(88, 88), is_train=True):
        self.size = size
        self.is_train = is_train
        self.enlarge_size = (int(size[0] * 1.1), int(size[1] * 1.1)) 

    def __call__(self, pil_images: List[Image.Image]) -> torch.Tensor:
        if self.is_train:
            do_flip = random.random() > 0.5
            # 降低旋轉角度，避免特徵變形過大
            angle = random.uniform(-2, 2)
            brightness = random.uniform(0.9, 1.1) 
            w_ext, h_ext = self.enlarge_size
            tw, th = self.size
            top = random.randint(0, h_ext - th)
            left = random.randint(0, w_ext - tw)
        else:
            do_flip = False
            angle = 0
            brightness = 1.0
            top, left = (self.enlarge_size[1]-self.size[1])//2, (self.enlarge_size[0]-self.size[0])//2
        
        t_frames = []
        for img in pil_images:
            if self.is_train:
                if do_flip: img = F.hflip(img)
                img = F.rotate(img, angle)
                img = F.adjust_brightness(img, brightness)
            img = F.resize(img, self.enlarge_size)
            img = F.crop(img, top, left, self.size[1], self.size[0])
            img_tensor = F.to_tensor(img)
            t_frames.append(img_tensor)
        return torch.stack(t_frames, dim=0)

class DMCLRVideoDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', seq_len: int = 40, 
                 channels: int = 1, shuffle_frames: bool = False):
        self.root = root
        self.split = split
        self.seq_len = seq_len
        self.channels = channels
        self.split_dir = os.path.join(root, split)
        self.aug = VideoAugmentation(size=(88, 88), is_train=(split == 'train'))
        
        if not os.path.exists(self.split_dir):
            self.samples = []
        else:
            entries = [os.path.join(self.split_dir, d) for d in os.listdir(self.split_dir)]
            self.samples = [p for p in entries if os.path.isdir(p)]
            self.samples.sort()
            
        self.label2idx = self._load_labels()

    def _load_labels(self) -> Optional[Dict[str, int]]:
        labels_path = os.path.join(os.path.dirname(__file__), 'labels.json')
        if not os.path.isfile(labels_path): return None
        with open(labels_path, 'r', encoding='utf-8') as f:
            return json.load(f).get('label2idx')

    def _pad_or_crop(self, frames: torch.Tensor):
        T, C, H, W = frames.shape
        if T >= self.seq_len:
            
            return frames[:self.seq_len]
        else:
            pad_count = self.seq_len - T
            last_frame = frames[-1:].repeat(pad_count, 1, 1, 1)
            return torch.cat([frames, last_frame], dim=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_dir = self.samples[idx]
        pil_images = []
        img_stack_path = os.path.join(sample_dir, 'images', 'img_stack.npy')
        if os.path.isfile(img_stack_path):
            arr = np.load(img_stack_path, allow_pickle=True)
            for i in range(arr.shape[0]):
                pil_images.append(Image.fromarray(arr[i].astype('uint8'), mode='L' if self.channels == 1 else 'RGB'))
        else:
            files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg'))])
            for f in files:
                with Image.open(os.path.join(sample_dir, f)) as img:
                    pil_images.append(img.convert('L' if self.channels == 1 else 'RGB'))
        
        frames = self.aug(pil_images)
        frames = self._pad_or_crop(frames)
        
        text = ""
        word_txt = os.path.join(sample_dir, 'word.txt')
        if os.path.isfile(word_txt):
            with open(word_txt, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        label = self.label2idx.get(text, -1) if self.label2idx else -1
        return {'frames': frames, 'label': label}