"""
Main execution script for Iris Recognition Pipeline.
"""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import cv2
from src.preprocessor import IrisPreprocessor
from src.feature_extractor import IrisFeatureExtractor
from src.matcher import IrisMatcher
from src import utils, config
from src.utils import save_output_image


def main(data_dir):
    base_dir = Path(data_dir)
    img_exts = {'.bmp', '.png', '.jpg', '.jpeg'}
    # Check that provided data directory exists early and give actionable hint
    if not base_dir.exists():
        print(f"Data directory not found: {base_dir}")
        alt = Path("./data")
        if alt.exists():
            print("Found './data' in the repository. Try: python main.py ./data")
        else:
            print("Please pass the dataset path, e.g. python main.py ./data")
        return

    # Initialize components
    preprocessor = IrisPreprocessor()
    extractor = IrisFeatureExtractor()
    matcher = IrisMatcher()

    # Load images
    image_paths = utils.list_all_images(base_dir, img_exts)
    print(f"Total images found: {len(image_paths)}")

    # Create Database
    database = []
    print("Processing images and building database...")
    
    for p in tqdm(image_paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        subj_id = utils.get_subject_id(p)
        file_base = f"{subj_id}_{p.stem}"

        # 1. Preprocess
        img_pre = preprocessor.preprocess(img)
        save_output_image(img_pre, f"{file_base}_1_preprocessed.png")

        # 2. Detect
        pupil, iris = preprocessor.detect_iris_circles(img_pre)
        if pupil is None or iris is None:
            continue

        # 3. Normalize
        norm_img = preprocessor.normalize_iris(img_pre, pupil, iris)

        # 4. Extract Features
        code_real, _, final_mask = extractor.extract_features(norm_img)

        # Add to DB
        subj_id = utils.get_subject_id(p)
        database.append({
            "id": subj_id,
            "path": str(p),
            "code_real": code_real,
            "mask": final_mask
        })

    print(f"Successfully processed: {len(database)} / {len(image_paths)}")

    # Visualization (Debug)
    print("Visualizing random samples...")
    utils.debug_iris_pipeline(preprocessor, database)

    # Matching Experiments
    genuine_scores = []
    impostor_scores = []
    num_data = len(database)

    print(f"Calculating Hamming Distances (Shift Range: {config.SHIFT_RANGE})...")
    
    for i in range(num_data):
        for j in range(i + 1, num_data):
            code1 = database[i]["code_real"]
            mask1 = database[i]["mask"]
            id1 = database[i]["id"]

            code2 = database[j]["code_real"]
            mask2 = database[j]["mask"]
            id2 = database[j]["id"]

            dist = matcher.compute_hamming_distance(
                code1, code2, mask1, mask2, shift_range=config.SHIFT_RANGE
            )

            if id1 == id2:
                genuine_scores.append(dist)
            else:
                impostor_scores.append(dist)

    print(f"Genuine pairs: {len(genuine_scores)}")
    print(f"Imposter pairs: {len(impostor_scores)}")

    # Evaluation
    labels = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    scores = np.concatenate([genuine_scores, impostor_scores])

    # ROC / EER Calculation
    # Note: Lower score is better, so we use negative scores for ROC
    fpr, tpr, thresholds = roc_curve(labels, -scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]

    print(f"Equal Error Rate (EER): {eer:.4f}")

    # Plot Results
    utils.plot_results(
        genuine_scores, impostor_scores, thresholds, fpr, tpr, eer, eer_idx
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris Recognition System")
    parser.add_argument("--data", type=str, default="./data/MMU-Iris-Database",
                        help="Path to the dataset directory")
    parser.add_argument("data_dir", nargs="?", default=None,
                        help="Optional positional path to dataset")
    args = parser.parse_args()

    # Prefer positional argument when provided (keeps backward compatibility)
    data_path = args.data_dir if args.data_dir is not None else args.data

    main(data_path)
