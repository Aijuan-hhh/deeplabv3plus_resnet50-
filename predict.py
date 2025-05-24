import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MoNuSegDataset
from models.deeplabv3plus_resnet50 import deeplabv3plus_resnet50
from matplotlib import pyplot as plt
import os


def calculate_metrics(pred_mask, true_mask):
    """
    计算 Dice系数 和 IoU
    Args:
        pred_mask (torch.Tensor): 预测掩码 [B, 1, H, W]，值在 [0,1] 之间
        true_mask (torch.Tensor): 真实掩码 [B, 1, H, W]，值在 {0, 1}
    Returns:
        dice (float): Dice系数
        iou (float): IoU
    """
    # 二值化预测掩码（阈值=0.5）
    pred_binary = (pred_mask > 0.5).float()

    # 计算交集和并集
    intersection = (pred_binary * true_mask).sum()
    union = (pred_binary + true_mask).sum() - intersection

    dice = (2. * intersection) / (pred_binary.sum() + true_mask.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    return dice.item(), iou.item()


def visualize_prediction(image, true_mask, pred_mask, save_path=None):
    """
    可视化输入图像、真实掩码和预测掩码
    Args:
        image (torch.Tensor): 输入图像 [C, H, W]
        true_mask (torch.Tensor): 真实掩码 [1, H, W]
        pred_mask (torch.Tensor): 预测掩码 [1, H, W]
        save_path (str): 图像保存路径（若为 None 则直接显示）
    """
    # 转换为 NumPy 并调整维度
    image_np = image.cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
    true_mask_np = true_mask.cpu().squeeze().numpy()  # [H, W]
    pred_mask_np = (pred_mask.cpu().squeeze() > 0.5).float().numpy()  # [H, W]

    plt.figure(figsize=(15, 5))

    # 输入图像
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Input Image")
    plt.axis("off")

    # 真实掩码
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask_np, cmap="gray")
    plt.title("True Mask")
    plt.axis("off")

    # 预测掩码
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask_np, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def test_model(model, device, test_data_path, batch_size=2, save_visualization=False):
    """
    测试模型性能
    Args:
        model (nn.Module): 加载权重的模型
        device (torch.device): 设备（CPU/GPU）
        test_data_path (str): 测试集路径
        batch_size (int): 批大小
        save_visualization (bool): 是否保存可视化结果
    """
    # 加载测试数据集
    test_dataset = MoNuSegDataset(test_data_path, image_size=(512, 512))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化统计量
    total_dice = 0.0
    total_iou = 0.0
    num_samples = 0

    # 设置为评估模式
    model.eval()

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # 前向传播
            outputs = model(images)
            pred_masks = torch.sigmoid(outputs)  # 转换为概率 [B, 1, H, W]

            # 计算指标
            batch_dice, batch_iou = calculate_metrics(pred_masks, masks)
            total_dice += batch_dice * images.size(0)
            total_iou += batch_iou * images.size(0)
            num_samples += images.size(0)

            # 可视化第一个样本
            if save_visualization and idx == 0:
                save_dir = "test_visualizations"
                os.makedirs(save_dir, exist_ok=True)
                for i in range(images.size(0)):
                    image = images[i]
                    true_mask = masks[i]
                    pred_mask = pred_masks[i]
                    save_path = os.path.join(save_dir, f"sample_{idx}_{i}.png")
                    visualize_prediction(image, true_mask, pred_mask, save_path)

    # 计算平均指标
    mean_dice = total_dice / num_samples
    mean_iou = total_iou / num_samples
    print(f"Test Results - Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    # 配置设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3plus_resnet50(num_classes=1)

    model_path = "best_deeplabv3plus.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model weights from {model_path}")
    model.to(device)

    test_data_path = "data/test"
    test_model(
        model,
        device,
        test_data_path,
        batch_size=2,
        save_visualization=True  # 是否保存可视化结果
    )