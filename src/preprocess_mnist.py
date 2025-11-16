import os
import pandas as pd
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm   # <-- progress bar

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "mnist_train.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# Create subfolders for each technique
subfolders = ["original", "resized", "normalized", "standardized", "augmented", "denoised"]
for sf in subfolders:
    os.makedirs(os.path.join(OUTPUT_DIR, sf), exist_ok=True)

def save_tensor_image(tensor, path):
    tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().numpy()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

def main():
    print("Loading CSV:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    to_pil = transforms.ToPILImage()

    resize_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    augment_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    denoise_transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor()
    ])

    # ----- only first 1000 images -----
    num_images = min(500, len(df))

    # Process first 1000 MNIST rows with a progress bar
    for idx in tqdm(range(num_images), desc="Processing MNIST images (first 500)"):
        row = df.iloc[idx].values
        label = int(row[0])
        pixels = row[1:].astype("float32")

        img_tensor = torch.tensor(pixels).view(1, 28, 28) / 255.0
        img_pil = to_pil(img_tensor)

        # Original
        save_tensor_image(
            img_tensor,
            os.path.join(OUTPUT_DIR, "original", f"image_{idx}_label_{label}.png")
        )

        # Resized
        img_resized = resize_transform(img_pil)
        save_tensor_image(
            img_resized,
            os.path.join(OUTPUT_DIR, "resized", f"image_{idx}_label_{label}.png")
        )

        # Normalized
        save_tensor_image(
            img_tensor,
            os.path.join(OUTPUT_DIR, "normalized", f"image_{idx}_label_{label}.png")
        )

        # Standardized
        mean = img_tensor.mean()
        std = img_tensor.std() if img_tensor.std() > 0 else 1.0
        img_standard = (img_tensor - mean) / std
        img_standard_disp = (img_standard - img_standard.min()) / (img_standard.max() - img_standard.min() + 1e-8)

        save_tensor_image(
            img_standard_disp,
            os.path.join(OUTPUT_DIR, "standardized", f"image_{idx}_label_{label}.png")
        )

        # Augmented
        img_aug = augment_transform(img_pil)
        save_tensor_image(
            img_aug,
            os.path.join(OUTPUT_DIR, "augmented", f"image_{idx}_label_{label}.png")
        )

        # Denoised
        img_denoised = denoise_transform(img_pil)
        save_tensor_image(
            img_denoised,
            os.path.join(OUTPUT_DIR, "denoised", f"image_{idx}_label_{label}.png")
        )

    print(f"Processed {num_images} MNIST images successfully!")

if __name__ == "__main__":
    main()
