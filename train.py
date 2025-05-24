import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
from models.deeplabv3plus_resnet50 import deeplabv3plus_resnet50  # 导入你的 DeepLabV3+ 模型
from dataset import MoNuSegDataset


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1.0
        pred = torch.sigmoid(pred)  # 将 logits 转换为概率
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice_coeff


def train_model(model, device, data_path, epochs=40, batch_size=2, lr=1e-4):
    # 加载数据集
    dataset = MoNuSegDataset(data_path, image_size=(512, 512))
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True, # 舍弃余数样本
                              num_workers=4,
                              pin_memory=True)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = DiceLoss()

    best_loss = float('inf')
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印当前 batch 的损失
            print(f"Epoch: {epoch + 1}/{epochs}, Batch Loss: {loss.item():.4f}")

        # 计算 epoch 平均损失
        epoch_loss_avg = epoch_loss / len(train_loader)
        train_losses.append(epoch_loss_avg)
        # 保存最佳模型
        if epoch_loss_avg < best_loss:
            best_loss = epoch_loss_avg
            torch.save(model.state_dict(), "best_deeplabv3plus.pth")
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DeepLabV3+ Training Loss")
    plt.legend()
    plt.savefig("deeplabv3plus_loss_curve.png")
    plt.close()

if __name__ == "__main__":
    # 配置设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3plus_resnet50(num_classes=1)  # 二分类任务（输出通道为1）
    model.to(device)
    data_path = "data/train"
    train_model(model, device, data_path, epochs=20, batch_size=2, lr=1e-4)