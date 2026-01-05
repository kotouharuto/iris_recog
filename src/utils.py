"""
Utility functions for file handling and visualization.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def list_all_images(base_dir: Path, extensions):
    """
    Recursively list all images in the base directory.
    """
    paths = []
    for p in base_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in extensions:
            paths.append(p)
    return sorted(paths)


def get_subject_id(path: Path):
    """
    Extract subject ID from file path structure.
    """
    parts = Path(path).parts
    subject_num = parts[-3]
    eye_side = parts[-2]
    return f"{subject_num}_{eye_side}"


def debug_iris_pipeline(preprocessor, database_or_paths, num_samples=3):
    """
    Visualize preprocessing, detection, and normalization steps.
    """
    import random
    
    if len(database_or_paths) > 0 and isinstance(database_or_paths[0], dict):
        paths = [d['path'] for d in database_or_paths]
    else:
        paths = database_or_paths

    sample_paths = random.sample(paths, min(len(paths), num_samples))

    for img_path in sample_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        plt.figure(figsize=(15, 4))
        plt.suptitle(f"File: {os.path.basename(img_path)}", fontsize=12)

        # 1. Preprocess
        img_pre = preprocessor.preprocess(img)
        plt.subplot(1, 4, 1)
        plt.imshow(img_pre, cmap='gray')
        plt.title("1. Preprocessed")
        plt.axis('off')

        # 2. Detection
        pupil, iris = preprocessor.detect_iris_circles(img)
        img_circles = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        plt.subplot(1, 4, 2)
        if pupil is not None and iris is not None:
            cv2.circle(img_circles, (int(iris[0]), int(iris[1])), int(iris[2]), (0, 255, 0), 2)
            cv2.circle(img_circles, (int(pupil[0]), int(pupil[1])), int(pupil[2]), (255, 0, 0), 2)
            plt.imshow(img_circles)
            plt.title("2. Detection Success")
        else:
            plt.imshow(img_circles)
            plt.title("2. Detection FAILED", color='red')
        plt.axis('off')

        # 3. Normalization
        plt.subplot(1, 4, 3)
        if pupil is not None and iris is not None:
            norm_img = preprocessor.normalize_iris(img, pupil, iris)
            plt.imshow(norm_img, cmap='gray')
            plt.title("3. Normalized")
        else:
            plt.text(0.5, 0.5, "No Data", ha='center')
        plt.axis('off')

        # 4. Feature Ready (Hist Eq)
        plt.subplot(1, 4, 4)
        if pupil is not None and iris is not None:
            norm_eq = cv2.equalizeHist(norm_img)
            plt.imshow(norm_eq, cmap='gray')
            plt.title("4. Feature Ready")
        else:
            plt.text(0.5, 0.5, "No Data", ha='center')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def plot_results(genuine_scores, impostor_scores, thresholds, fpr, tpr, eer, eer_idx):
    """
    Plot score distribution, FAR/FRR, and ROC curves.
    """
    # Score Distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(genuine_scores, bins=30, alpha=0.7, label='Genuine', density=True, edgecolor='black')
    plt.hist(impostor_scores, bins=30, alpha=0.7, label='Imposter', density=True, edgecolor='black')
    plt.axvline(x=-thresholds[eer_idx], color='r', linestyle='--', label='EER Threshold')
    plt.xlabel('Hamming Distance')
    plt.title('Score Distribution')
    plt.legend()

    # ROC Curve
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[eer_idx], tpr[eer_idx], color='red', label=f'EER point ({eer:.2f})')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def save_output_image(image, filename):
    """
    Saves the specified image to iris_recog/outputs/images with an absolute path.
    """
    # Save under repository root: <repo>/outputs/images
    base_dir = Path(__file__).resolve().parents[1] / "outputs" / "images"

    # Create directory if missing
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {base_dir}")

    save_path = base_dir / filename

    # Prepare image for saving: ensure uint8 and proper scaling
    img = image
    if isinstance(img, np.ndarray):
        if np.issubdtype(img.dtype, np.floating):
            max_val = img.max() if img.size > 0 else 0
            if max_val <= 1.0:
                img = (img * 255.0)
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        try:
            img = np.array(img)
            if np.issubdtype(img.dtype, np.floating):
                max_val = img.max() if img.size > 0 else 0
                if max_val <= 1.0:
                    img = (img * 255.0)
            img = np.clip(img, 0, 255).astype(np.uint8)
        except Exception:
            raise ValueError("Unsupported image type for saving")

    success = cv2.imwrite(str(save_path), img)
    if success:
        print(f"Saved: {save_path}")
    else:
        print(f"Failed to save image: {save_path}")