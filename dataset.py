import torch
import cv2
import os
import glob

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class MoNuSegDataset(Dataset):
    def __init__(self, data_path, image_size=(512, 512)):
        """
        :param data_path: 根目录，下面有 image/ 和 label/
        :param image_size: resize 图像大小，适应 DeepLabv3+
        """
        self.data_path = data_path
        self.image_size = image_size
        self.image_paths = glob.glob(os.path.join(data_path, 'Images', '*.tif'))
        self.image_paths.sort()  # 确保图像和标签匹配
        self.label_paths = [p.replace('Images', 'Masks').replace('.tif','.png') for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def augment(self, img, mask):
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            img = cv2.flip(img, flipCode)
            mask = cv2.flip(mask, flipCode)
        return img, mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.label_paths[idx]

        # 读取图像(.tif)
        img = np.array(Image.open(img_path).convert('RGB'))  # 转为RGB以保证3通道
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize 到统一大小
        img = cv2.resize(img, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # 数据增强
        img, mask = self.augment(img, mask)

        # 转 tensor 并归一化
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC -> CHW
        mask = (mask > 127).astype(np.uint8)  # 0 or 1
        mask = torch.from_numpy(mask).long().unsqueeze(0)  # CHW (1, H, W)

        return img, mask

# 示例用法
if __name__ == "__main__":
    dataset = MoNuSegDataset(r"data\train", image_size=(512, 512))
    print("数据量：", len(dataset))

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for images, masks in loader:
        print(f"图像形状: {images.shape}")  # [B, 3, H, W]
        print(f"标签形状: {masks.shape}")  # [B, 1, H, W]
        break

    image_tensor, mask_tensor = dataset[0]  # image: [3, H, W], mask: [1, H, W]

    image_np = image_tensor.permute(1, 2, 0).numpy()  # [H, W, 3]
    mask_np = mask_tensor.squeeze().numpy()  # [H, W]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Label Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
