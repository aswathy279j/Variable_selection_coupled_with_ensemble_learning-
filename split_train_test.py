import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_and_save_csv(csv_file, test_size=0.2, random_state=42):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Define features and target (optional, but you used it)
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values

    # Split indices while preserving structure
    train_idx, test_idx = train_test_split(
        range(len(data)),
        test_size=test_size,
        stratify=y,
        shuffle=True,
        random_state=random_state
    )

    # Create train and test DataFrames
    train_df = data.iloc[train_idx]
    test_df = data.iloc[test_idx]

    # Build file names
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    train_file = f"{base_filename}_train.xlsx"
    test_file = f"{base_filename}_test.xlsx"

    # Save to Excel
    train_df.to_excel(train_file, index=False)
    test_df.to_excel(test_file, index=False)

    # Confirmation messages
    print(f"Train split saved as: {train_file}")
    print(f"Test split saved as: {test_file}")

    return train_file, test_file


# usage:
if __name__ == "__main__":
    split_and_save_csv("Sunflower.csv", test_size=0.2)
    