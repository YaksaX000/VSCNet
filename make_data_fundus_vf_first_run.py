import os
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import savemat


JSON_PATH = "/media/storage/eye/TSGH/Ophthal1117_processed_V3/matched_data/FundusImages_VF_match_60.json"
IMG_ROOT  = "/media/storage/eye/TSGH/fundus_processed/images"
OUT_DIR   = str(Path.home() / "projects" / "VSCNet" / "data_fundus_vf")


# take 1st
ONLY_USE_INDEX_1 = True

# train/test split
TEST_PID_RATIO = 0.2

# semantic feature .mat 設定（對應 md_label 0,1,2,3 => 4 classes）
NUM_CLASSES = 4
EMB_DIM = 300
RANDOM_SEED = 0


def build_image_path(pid: str, eye: str, yyyymmdd: str) -> str:
    # 檔名規則：L_20190430_1.jpg / R_20190430_1.jpg
    if ONLY_USE_INDEX_1:
        fname = f"{eye}_{yyyymmdd}_1.jpg"
        return os.path.join(IMG_ROOT, pid, fname)
    else:
        raise NotImplementedError


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(JSON_PATH, "r") as f:
        j = json.load(f)

    data = j["data"]

    # 1) 收集所有 pid，做 pid-level split
    pids = sorted(list(data.keys()))
    random.seed(RANDOM_SEED)
    random.shuffle(pids)

    n_test = int(len(pids) * TEST_PID_RATIO)
    test_pids = set(pids[:n_test])
    train_pids = set(pids[n_test:])

    train_pairs = []
    test_pairs = []

    missing_count = 0
    total_count = 0

    # 2)pid -> eye -> date
    for pid, pid_dict in data.items():
        for eye, eye_dict in pid_dict.items():
            
            for visit_date, rec in eye_dict.items():
                total_count += 1
                img_path = build_image_path(pid, eye, visit_date)
                if not os.path.exists(img_path):
                    missing_count += 1
                    continue

                md_label = rec.get("md_label", None)
                if md_label is None:
                    continue

                # 確保 label 在 0~3
                if not (0 <= int(md_label) < NUM_CLASSES):
                    continue

                # ---- 取 4 維 clinical semantic ----
                vfi = rec.get("vfi", None)
                md  = rec.get("md", None)
                psd = rec.get("psd", None)
                ght = rec.get("ght", None)

                # 若缺值就跳過（今晚先求穩）
                if any(x is None for x in [vfi, md, psd, ght]):
                    continue

                # 轉成 float（避免有字串）
                vfi = float(vfi)
                md  = float(md)
                psd = float(psd)
                ght = float(ght)

                if pid in test_pids:
                    test_pairs.append((img_path, int(md_label), vfi, md, psd, ght))
                else:
                    train_pairs.append((img_path, int(md_label), vfi, md, psd, ght))

    # 3) 寫入 txt（每行一張圖對一個 label）
    train_img_txt = os.path.join(OUT_DIR, "train_images.txt")
    train_lab_txt = os.path.join(OUT_DIR, "train_labels.txt")
    test_img_txt  = os.path.join(OUT_DIR, "test_images.txt")
    test_lab_txt  = os.path.join(OUT_DIR, "test_labels.txt")

    with open(train_img_txt, "w") as f_img, open(train_lab_txt, "w") as f_lab:
        for p, y, vfi, md, psd, ght in train_pairs:
            f_img.write(f"{p}\t{vfi}\t{md}\t{psd}\t{ght}\n")
            f_lab.write(str(y) + "\n")

    with open(test_img_txt, "w") as f_img, open(test_lab_txt, "w") as f_lab:
        for p, y, vfi, md, psd, ght in test_pairs:
            f_img.write(f"{p}\t{vfi}\t{md}\t{psd}\t{ght}\n")
            f_lab.write(str(y) + "\n")

    # 4) 生成 ingredient_all_feature.mat
    np.random.seed(RANDOM_SEED)
    feat = np.random.randn(NUM_CLASSES, EMB_DIM).astype(np.float32)

    mat_path = os.path.join(OUT_DIR, "ingredient_all_feature.mat")
    savemat(mat_path, {"ingredient_all_feature": feat})

    # Report
    print("=== DONE ===")
    print("OUT_DIR:", OUT_DIR)
    print("train samples:", len(train_pairs))
    print("test samples :", len(test_pairs))
    print("missing images skipped:", missing_count, "/", total_count)
    print("mat shape:", feat.shape)
    print("Example train_images.txt head:")
    if len(train_pairs) > 0:
        print(" ", train_pairs[0][0], "label=", train_pairs[0][1])


if __name__ == "__main__":
    main()