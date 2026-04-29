import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

from model import (
    extract_color_features,
    extract_texture_features,
    extract_shape_features
)

DATASET_DIR = "FNNPK"
def extract_dataset_features():
    feats = []
    paths = []
    labels = []

    for cls in sorted(os.listdir(DATASET_DIR)):
        cls_path = os.path.join(DATASET_DIR, cls)

        if not os.path.isdir(cls_path):
            continue

        for f in os.listdir(cls_path):
            if not f.lower().endswith((".jpg",".png",".jpeg")):
                continue

            img_path = os.path.join(cls_path, f)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224,224))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            color = extract_color_features(img)
            texture = extract_texture_features(gray)
            shape = extract_shape_features(gray)

            feat = np.concatenate([color, texture, shape])

            feats.append(feat)
            paths.append(img_path)
            labels.append(cls)

    return np.array(feats), paths, labels


# Extract features of query image
def extract_query_features(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color = extract_color_features(img)
    texture = extract_texture_features(gray)
    shape = extract_shape_features(gray)

    return np.concatenate([color, texture, shape]), img


# Retrieve top-3 similar
def retrieve_top3(query_path):
    dataset_feats, dataset_paths, dataset_labels = extract_dataset_features()
    q_feat, q_img = extract_query_features(query_path)

    sims = cosine_similarity([q_feat], dataset_feats)[0]

    top3 = sims.argsort()[-3:][::-1]

    print("\nTop-3 similar images:")
    for idx in top3:
        print(f"{dataset_paths[idx]} → class={dataset_labels[idx]}  (sim={sims[idx]:.4f})")

    # ---- Create one combined image using OpenCV ----
    result = cv2.resize(q_img, (300,300))
    text_pos = (10, 25)
    cv2.putText(result, "Query Image", text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,0,255), 2)

    # Append top-3
    for i, idx in enumerate(top3):
        img = cv2.imread(dataset_paths[idx])
        img = cv2.resize(img, (300,300))
        cv2.putText(img, dataset_labels[idx],
                    (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,0,0), 2)

        result = np.hstack((result, img))

    # Save combined output
    cv2.imwrite("cbir_result.png", result)
    print("\nSaved result as cbir_result.png")


if __name__ == "__main__":
    p = input("Enter query image path: ")
    retrieve_top3(p)
