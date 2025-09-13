import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import glob
import cv2
from tqdm import tqdm
import torch
from model_DBPF import PVTFormer,AttUNet
from utils import create_dir, seeding
from utils import calculate_metrics
from train_DBPF import load_data

def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)  # Ensure y_pred is (H, W, 3)
    return y_pred

def print_score(metrics_score, num_samples):
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples
    acc = metrics_score[4] / num_samples
    f2 = metrics_score[5] / num_samples
    hd = metrics_score[6] / num_samples

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - HD: {hd:1.4f}")

def evaluate(model, unet_model, save_path, test_x, test_y, size):
    metrics_score = [0.0] * 7  # Initialize metrics
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = x.split("/")
        name = f"{name[-3]}_{name[-1]}"

        # Load and preprocess image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0) / 255.0
        image = torch.from_numpy(image).float().to(device)

        # Load and preprocess mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        mask = torch.from_numpy(mask).to(device)

        with torch.no_grad():
            # Calculate UNet output
            unet_output = unet_model(image)

            # Calculate PVTFormer output
            start_time = time.time()
            y_pred = model(image, unet_output)  # Pass UNet output to PVTFormer
            y_pred = torch.sigmoid(y_pred)
            end_time = time.time() - start_time
            time_taken.append(end_time)

            # Evaluation metrics
            score = calculate_metrics(mask, y_pred)
            metrics_score = list(map(add, metrics_score, score))

            # Process predicted mask
            y_pred_mask = process_mask(y_pred)

        # Convert save_mask to 3D for concatenation
        save_mask = np.expand_dims(save_mask, axis=-1)  # Shape (H, W, 1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)  # Shape (H, W, 3)

        # Ensure line is 3D
        line = np.ones((size[0], 10, 3), dtype=np.uint8) * 255  # Shape (H, W, 3)

        # Save the image - mask - pred
        cat_images = np.concatenate([save_img, line, save_mask, line, y_pred_mask], axis=1)
        cv2.imwrite(f"{save_path}/joint/{name}", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}", y_pred_mask)

    print_score(metrics_score, len(test_x))
    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken
    print("Mean FPS: ", mean_fps)

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Load the checkpoint """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    unet_model = AttUNet().to(device)  # Create UNet instance
    model = PVTFormer(unet_model).to(device)  # Create PVTFormer instance with UNet
    checkpoint_path = "files(PVT_DBPF_liver)/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "../liver/liver"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    save_path = f"results_mine_liver/"
    for item in ["mask", "joint"]:
        create_dir(f"{save_path}/{item}")

    size = (448, 448)
    evaluate(model, unet_model, save_path, test_x, test_y, size)