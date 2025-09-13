import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model_DBPF import PVTFormer,AttUNet
from metrics import DiceLoss, DiceBCELoss

def load_data(path):
    def get_data(path):
        # 打印完整路径，帮助调试
        print(f"Searching images in: {os.path.join(path, 'images', 'liver_*.jpg')}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Full path being searched: {os.path.abspath(path)}")

        # 调试：打印实际搜索路径
        image_search_path = os.path.join(path, "images", "liver_*.jpg")
        label_search_path = os.path.join(path, "labels", "liver_*.jpg")
        
        print(f"Image search path: {image_search_path}")
        print(f"Label search path: {label_search_path}")

        # 获取所有图像文件
        images = sorted(glob.glob(image_search_path))
        labels = sorted(glob.glob(label_search_path))
        
        # 打印详细的文件信息
        print("Image files found:")
        for img in images[:10]:  # 打印前10个图像路径
            print(img)
        
        print("\nLabel files found:")
        for label in labels[:10]:  # 打印前10个标签路径
            print(label)

        # 检查图像和标签数量是否匹配
        if len(images) != len(labels):
            print(f"Warning: Mismatch in images and labels count. Images: {len(images)}, Labels: {len(labels)}")
        
        return images, labels

    try:
        """ 获取所有数据 """
        all_x, all_y = get_data(path)

        print(f"Total images found: {len(all_x)}")
        print(f"Total labels found: {len(all_y)}")

        # 如果没有找到数据，直接返回空列表
        if len(all_x) == 0:
            raise ValueError(f"No images found in path: {path}")

        """ 随机打乱数据 """
        np.random.seed(42)
        indices = np.random.permutation(len(all_x))
        all_x = np.array(all_x)[indices]
        all_y = np.array(all_y)[indices]

        """ 按1:1:2比例划分数据集 """
        total_samples = len(all_x)
        train_end = int(total_samples * 0.25)  
        valid_end = int(total_samples * 0.5)  

        train_x = all_x[:train_end]
        train_y = all_y[:train_end]

        valid_x = all_x[train_end:valid_end]
        valid_y = all_y[train_end:valid_end]

        test_x = all_x[valid_end:]
        test_y = all_y[valid_end:]

        return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

    except Exception as e:
        print(f"Error in load_data: {e}")
        # 打印详细的异常信息
        import traceback
        traceback.print_exc()
        return [([], []), ([], []), ([], [])]

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        return image, mask

    def __len__(self):
        return self.n_samples

def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        # 计算 UNet 的输出
        unet_output = unet_model(x)  # 计算 UNet 输出

        # 传递 UNet 输出到 PVTFormer
        y_pred = model(x, unet_output)  # 传递 unet_output
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        y_pred = torch.sigmoid(y_pred)
        for yt, yp in zip(y, y_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            # 计算 UNet 的输出
            unet_output = unet_model(x)  # 计算 UNet 输出

            # 传递 UNet 输出到 PVTFormer
            y_pred = model(x, unet_output)  # 传递 unet_output

            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            y_pred = torch.sigmoid(y_pred)
            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files(PVT_DBPF_liver)")

    """ Training logfile """
    train_log_path = "files(PVT_DBPF_liver)/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files(PVT_DBPF_liver)/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 4
    num_epochs = 200
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = "files(PVT_DBPF_liver)/checkpoint.pth"
    path = "../liver/liver"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 创建 UNet 实例
    unet_model = AttUNet().to(device)  # 创建 UNet 实例

    # 创建 PVTFormer 实例并传递 unet_model
    model = PVTFormer(unet_model).to(device)  # 传递 UNet 实例

    model = model.to(device)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print(f"UNet device: {next(unet_model.parameters()).device}")
    print(f"PVTFormer device: {next(model.parameters()).device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
