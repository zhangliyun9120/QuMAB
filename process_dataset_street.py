from collections import Counter
import os
import pandas as pd
import shutil
import json
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit



def process_filename(filename):
    return re.sub(r'\((\d+)\)', r'_\1', filename)


def split_dataset_normal():
    df = pd.read_csv('datasets/annotated/original/labels.csv')

    df['name'] = df['name'].apply(process_filename)

    print(df.head())
    print(df.columns)

    # normalize_label
    # -3 to 0
    # -2 to 1
    # -1 to 2
    # 1 to 3
    # 2 to 4
    # 3 to 5
    def normalize_label(x):
        mapping = {-3: 0, -2: 1, -1: 2, 1: 3, 2: 4, 3: 5}
        return mapping.get(x, x)

    label_columns = [
        'Happiness_1', 'Healthy_1', 'Safe_1', 'Lively_1', 'Orderly_1',
        'Happiness_2', 'Healthy_2', 'Safe_2', 'Lively_2', 'Orderly_2',
        'Happiness_3', 'Healthy_3', 'Safe_3', 'Lively_3', 'Orderly_3',
        'Happiness_4', 'Healthy_4', 'Safe_4', 'Lively_4', 'Orderly_4',
        'Happiness_5', 'Healthy_5', 'Safe_5', 'Lively_5', 'Orderly_5',
        'Happiness_6', 'Healthy_6', 'Safe_6', 'Lively_6', 'Orderly_6',
        'Happiness_7', 'Healthy_7', 'Safe_7', 'Lively_7', 'Orderly_7',
        'Happiness_8', 'Healthy_8', 'Safe_8', 'Lively_8', 'Orderly_8',
        'Happiness_9', 'Healthy_9', 'Safe_9', 'Lively_9', 'Orderly_9',
        'Happiness_10', 'Healthy_10', 'Safe_10', 'Lively_10', 'Orderly_10'
    ]

    # Apply normalization to label columns
    for col in label_columns:
        df[f'{col}_normalized'] = df[col].apply(normalize_label).astype(int)

    print(df[[f'{col}_normalized' for col in label_columns]].head())

    # Prepare features and labels
    X = df['name']
    y = df[[f'{col}_normalized' for col in label_columns]]

    # Simple random split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Remove '_normalized' suffix from column names
    y_train.columns = [col.replace('_normalized', '') for col in y_train.columns]
    y_val.columns = [col.replace('_normalized', '') for col in y_val.columns]
    y_test.columns = [col.replace('_normalized', '') for col in y_test.columns]

    # Concatenate features and labels for each split
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Save datasets
    train_data.to_csv('datasets/annotated/normal/train_data.csv', index=False)
    val_data.to_csv('datasets/annotated/normal/val_data.csv', index=False)
    test_data.to_csv('datasets/annotated/normal/test_data.csv', index=False)

    print(f"\nDataset split completed:")
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")

    # Verify saved data
    saved_data = pd.read_csv('datasets/annotated/normal/train_data.csv')
    print(f"Saved training data shape: {saved_data.shape}")


if __name__ == "__main__":

    split_dataset_normal()