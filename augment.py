import pandas as pd
import os

# Read the data
data = pd.read_csv("data/train_val_cleaned.csv")

attr = [
    "unnecessary",
    "mandatory",
    "pharma",
    "conspiracy",
    "political",
    "country",
    "rushed",
    "ingredients",
    "side-effect",
    "ineffective",
    "religious",
    "none",
]

def process_attribute(attribute):
    df_attr = data.copy()
    df_attr[attribute] = df_attr["labels"].apply(lambda x: 1 if attribute in x else 0)

    # Separate the rows with the attribute and without the attribute
    df_attr_1 = df_attr[df_attr[attribute] == 1]
    df_attr_0 = df_attr[df_attr[attribute] == 0]

    # Determine the number of rows required to balance the dataset
    num_rows_1 = len(df_attr_1)
    num_rows_0 = len(df_attr_0)

    # Either oversample the minority class or undersample the majority class to achieve balance
    if num_rows_1 > num_rows_0:
        df_attr_1 = df_attr_1.sample(num_rows_0, replace=False)
    else:
        df_attr_0 = df_attr_0.sample(num_rows_1, replace=False)

    # Concatenate the balanced datasets
    df_attr_balanced = pd.concat([df_attr_1, df_attr_0])
    df_attr_balanced = df_attr_balanced.drop(columns=["labels"])
    df_attr_balanced = df_attr_balanced.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset

    print(df_attr_balanced.head())
    print(df_attr_balanced[attribute].value_counts())
    df_attr_balanced.to_csv(f"data/df_{attribute}.csv", index=False)

# Apply the process to all attributes
for attribute in attr:
    process_attribute(attribute)

# Delete all files in the folder "/model"
folder = "models"
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print("Failed to delete %s. Reason: %s" % (file_path, e))
