"""
Synthetic Image Dataset Generator
==================================
This script generates synthetic forged image dataset.

Author: Student Project Team
Date: 2024
Course: Final Year Major Project
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd


def create_base_images(n_images=250, output_dir='raw/original', size=(512, 512)):
    """
    Create base original images.

    Args:
        n_images (int): Number of images to create
        output_dir (str): Output directory
        size (tuple): Image size

    Returns:
        list: List of created image paths
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {n_images} base images...")
    image_paths = []

    for i in range(n_images):
        # Create random colored image
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        img = Image.new('RGB', size, color=color)
        draw = ImageDraw.Draw(img)

        # Add some shapes
        n_shapes = random.randint(3, 8)
        for _ in range(n_shapes):
            shape_type = random.choice(['rectangle', 'circle', 'line'])

            x1 = random.randint(0, size[0] - 100)
            y1 = random.randint(0, size[1] - 100)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 150)

            shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            if shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=shape_color, outline='black')
            elif shape_type == 'circle':
                draw.ellipse([x1, y1, x2, y2], fill=shape_color, outline='black')
            else:
                draw.line([x1, y1, x2, y2], fill=shape_color, width=3)

        # Add some text
        text = f"Image {i+1}"
        draw.text((10, 10), text, fill='white')

        # Save image
        img_path = os.path.join(output_dir, f'original_{i+1:04d}.png')
        img.save(img_path)
        image_paths.append(img_path)

    print(f"  Created {n_images} original images")
    return image_paths


def apply_copy_move_forgery(image):
    """
    Apply copy-move forgery to an image.

    Args:
        image (PIL.Image): Input image

    Returns:
        PIL.Image: Forged image
    """
    img = image.copy()
    width, height = img.size

    # Select a random region to copy
    region_w = random.randint(50, 150)
    region_h = random.randint(50, 150)

    # Source position
    src_x = random.randint(0, width - region_w)
    src_y = random.randint(0, height - region_h)

    # Destination position (different from source)
    dst_x = random.randint(0, width - region_w)
    dst_y = random.randint(0, height - region_h)

    # Ensure destination is different
    while abs(dst_x - src_x) < 30 and abs(dst_y - src_y) < 30:
        dst_x = random.randint(0, width - region_w)
        dst_y = random.randint(0, height - region_h)

    # Copy region
    region = img.crop((src_x, src_y, src_x + region_w, src_y + region_h))

    # Paste to destination
    img.paste(region, (dst_x, dst_y))

    return img


def apply_splicing_forgery(image1, image2):
    """
    Apply splicing forgery (combine parts from two images).

    Args:
        image1 (PIL.Image): First image
        image2 (PIL.Image): Second image

    Returns:
        PIL.Image: Forged image
    """
    # Resize image2 to match image1 if needed
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    img = image1.copy()
    width, height = img.size

    # Take a region from image2
    region_w = random.randint(100, 200)
    region_h = random.randint(100, 200)
    x = random.randint(0, width - region_w)
    y = random.randint(0, height - region_h)

    region = image2.crop((x, y, x + region_w, y + region_h))

    # Paste onto image1
    paste_x = random.randint(0, width - region_w)
    paste_y = random.randint(0, height - region_h)
    img.paste(region, (paste_x, paste_y))

    return img


def apply_text_manipulation(image):
    """
    Manipulate text in an image.

    Args:
        image (PIL.Image): Input image

    Returns:
        PIL.Image: Image with manipulated text
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Add fake/misleading text
    fake_texts = [
        "FAKE NEWS",
        "NOT REAL",
        "MANIPULATED",
        "EDITED IMAGE",
        "DOCTORED",
        "ALTERED"
    ]

    text = random.choice(fake_texts)

    # Random position
    x = random.randint(10, img.size[0] - 150)
    y = random.randint(10, img.size[1] - 50)

    # Draw text with background
    draw.rectangle([x-5, y-5, x+len(text)*8, y+25], fill='yellow')
    draw.text((x, y), text, fill='red')

    return img


def apply_noise_addition(image):
    """
    Add noise to image (simple forgery technique).

    Args:
        image (PIL.Image): Input image

    Returns:
        PIL.Image: Noisy image
    """
    img_array = np.array(image).astype(np.float32)

    # Add Gaussian noise
    noise = np.random.normal(0, 25, img_array.shape)
    noisy = img_array + noise

    # Clip values
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy)


def create_forged_images(original_paths, n_forged=250, output_dir='raw/forged'):
    """
    Create forged versions of original images.

    Args:
        original_paths (list): Paths to original images
        n_forged (int): Number of forged images to create
        output_dir (str): Output directory

    Returns:
        list: List of forged image paths
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {n_forged} forged images...")
    forged_paths = []

    for i in range(n_forged):
        # Select random original image(s)
        orig_path = random.choice(original_paths)
        img = Image.open(orig_path)

        # Apply random forgery technique
        forgery_type = random.choice(['copy_move', 'splicing', 'text_manipulation', 'noise'])

        if forgery_type == 'copy_move':
            forged = apply_copy_move_forgery(img)
        elif forgery_type == 'splicing':
            # Need another image for splicing
            other_path = random.choice(original_paths)
            other_img = Image.open(other_path)
            forged = apply_splicing_forgery(img, other_img)
        elif forgery_type == 'text_manipulation':
            forged = apply_text_manipulation(img)
        else:  # noise
            forged = apply_noise_addition(img)

        # Save forged image
        forged_path = os.path.join(output_dir, f'forged_{i+1:04d}.png')
        forged.save(forged_path)
        forged_paths.append(forged_path)

    print(f"  Created {n_forged} forged images")
    return forged_paths


def create_dataset_csv(original_paths, forged_paths, output_path='image_dataset.csv'):
    """
    Create CSV file for the image dataset.

    Args:
        original_paths (list): Paths to original images
        forged_paths (list): Paths to forged images
        output_path (str): Output CSV path

    Returns:
        pd.DataFrame: Dataset dataframe
    """
    data = {
        'image_path': original_paths + forged_paths,
        'label': [0] * len(original_paths) + [1] * len(forged_paths)
    }

    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df.to_csv(output_path, index=False)

    print(f"\nDataset CSV created: {output_path}")
    print(f"  - Total images: {len(df)}")
    print(f"  - Authentic (label=0): {len(original_paths)}")
    print(f"  - Forged (label=1): {len(forged_paths)}")

    return df


def split_dataset(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split image dataset into train, validation, and test sets.

    Args:
        csv_path (str): Path to full dataset CSV
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
    """
    print("\nSplitting image dataset...")

    df = pd.read_csv(csv_path)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate sizes
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Save splits
    base_path = os.path.dirname(csv_path)
    train_df.to_csv(os.path.join(base_path, 'train_images.csv'), index=False)
    val_df.to_csv(os.path.join(base_path, 'val_images.csv'), index=False)
    test_df.to_csv(os.path.join(base_path, 'test_images.csv'), index=False)

    print(f"  - Training set: {len(train_df)} images")
    print(f"  - Validation set: {len(val_df)} images")
    print(f"  - Test set: {len(test_df)} images")


def main():
    """Main function to generate complete synthetic image dataset."""
    print("=" * 50)
    print("GENERATING SYNTHETIC IMAGE DATASET")
    print("=" * 50)

    # Create directories
    os.makedirs('raw', exist_ok=True)
    os.makedirs('processed', exist_ok=True)

    # Create original images
    original_paths = create_base_images(
        n_images=250,
        output_dir='raw/original',
        size=(256, 256)  # Smaller for faster processing
    )

    # Create forged images
    forged_paths = create_forged_images(
        original_paths,
        n_forged=250,
        output_dir='raw/forged'
    )

    # Create dataset CSV
    df = create_dataset_csv(
        original_paths,
        forged_paths,
        output_path='processed/image_dataset.csv'
    )

    # Split dataset
    split_dataset('processed/image_dataset.csv')

    print("\n" + "=" * 50)
    print("[SUCCESS] Synthetic image dataset generation completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
