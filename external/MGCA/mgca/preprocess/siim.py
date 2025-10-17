import os

import numpy as np
import pandas as pd
from mgca.constants import (
    CHEXPERT_COMPETITION_TASKS,
    CHEXPERT_DATA_DIR,
    CHEXPERT_PATH_COL,
    CHEXPERT_TEST_CSV,
    CHEXPERT_TRAIN_CSV,
    CHEXPERT_UNCERTAIN_MAPPINGS,
    CHEXPERT_VALID_CSV,
    CHEXPERT_VIEW_COL,
    COVIDX_DATA_DIR,
    COVIDX_TEST_CSV,
    COVIDX_TRAIN_CSV,
    COVIDX_VALID_CSV,
    MIMIC_CXR_DATA_DIR,
    MIMIC_CXR_PATH_COL,
    MIMIC_CXR_TEST_CSV,
    MIMIC_CXR_TRAIN_CSV,
    MIMIC_CXR_VALID_CSV,
    MIMIC_CXR_VIEW_COL,
    PNEUMOTHORAX_IMG_DIR,
    PNEUMOTHORAX_ORIGINAL_TRAIN_CSV,
    PNEUMOTHORAX_TEST_CSV,
    PNEUMOTHORAX_TRAIN_CSV,
    PNEUMOTHORAX_VALID_CSV,
    RSNA_DATA_DIR,
    RSNA_IMG_DIR,
    RSNA_TEST_CSV,
    RSNA_TRAIN_CSV,
    RSNA_VALID_CSV,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(0)


def preprocess_pneumothorax_data(test_fac=0.15):
    # try:
    df = pd.read_csv(PNEUMOTHORAX_ORIGINAL_TRAIN_CSV)
    # except:
    #     raise Exception(
    #         "Please make sure the the SIIM Pneumothorax dataset is \
    #         stored at {PNEUMOTHORAX_DATA_DIR}"
    #     )

    # get image paths
    os.listdir(PNEUMOTHORAX_IMG_DIR)
    img_paths = {}
    for subdir, dirs, files in tqdm(os.walk(PNEUMOTHORAX_IMG_DIR)):
        for f in files:
            if "dcm" in f:
                # remove dcm
                file_id = f[:-4]
                img_paths[file_id] = os.path.join(subdir[105:], f)

    # no encoded pixels mean healthy
    df["Label"] = df.apply(
        lambda x: 0.0 if x[" EncodedPixels"] == " -1" else 1.0, axis=1
    )
    df["Path"] = df["ImageId"].apply(lambda x: img_paths[x])

    # split data
    train_df, test_val_df = train_test_split(df, test_size=test_fac * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Label"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Label"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Label"].value_counts())

    train_df.to_csv(PNEUMOTHORAX_TRAIN_CSV)
    valid_df.to_csv(PNEUMOTHORAX_VALID_CSV)
    test_df.to_csv(PNEUMOTHORAX_TEST_CSV)


if __name__ == "__main__":
    preprocess_pneumothorax_data()
