import pandas as pd
import os

# Read the data
data = pd.read_csv("data/train_val.csv")

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

factor = 6

df_unnecessary = data.copy()
df_unnecessary["unnecessary"] = df_unnecessary["labels"].apply(
    lambda x: 1 if "unnecessary" in x else 0
)

# Filter rows where 'unnesecary' column is 1
data_fil = df_unnecessary[df_unnecessary["unnecessary"] == 1]

# repeat the rows  2 * factor times
data_fil = data_fil.loc[data_fil.index.repeat(2 * factor)]

# Add the new rows to the original data
df_unnecessary = pd.concat([df_unnecessary, data_fil])


df_unnecessary = df_unnecessary.drop(columns=["labels"])

print(df_unnecessary.head())

# print the number of 1s in the column unnsecessary
print(df_unnecessary["unnecessary"].value_counts())
# Save the new data
#shuffle the dataset 
df_unnecessary = df_unnecessary.sample(frac=1).reset_index(drop=True)
df_unnecessary.to_csv("data/df_unnecessary.csv", index=False)


# Repeat the same for the other attributes
df_mandatory = data.copy()
df_mandatory["mandatory"] = df_mandatory["labels"].apply(
    lambda x: 1 if "mandatory" in x else 0
)
data_fil = df_mandatory[df_mandatory["mandatory"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(2 * factor)]
df_mandatory = pd.concat([df_mandatory, data_fil])
df_mandatory = df_mandatory.drop(columns=["labels"])
print(df_mandatory.head())
print(df_mandatory["mandatory"].value_counts())
df_mandatory = df_mandatory.sample(frac=1).reset_index(drop=True)
df_mandatory.to_csv("data/df_mandatory.csv", index=False)

df_pharma = data.copy()
df_pharma["pharma"] = df_pharma["labels"].apply(lambda x: 1 if "pharma" in x else 0)
data_fil = df_pharma[df_pharma["pharma"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(factor)]
df_pharma = pd.concat([df_pharma, data_fil])
df_pharma = df_pharma.drop(columns=["labels"])
print(df_pharma.head())
print(df_pharma["pharma"].value_counts())
df_pharma.to_csv("data/df_pharma.csv", index=False)

df_conspiracy = data.copy()
df_conspiracy["conspiracy"] = df_conspiracy["labels"].apply(
    lambda x: 1 if "conspiracy" in x else 0
)
data_fil = df_conspiracy[df_conspiracy["conspiracy"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(20)]
df_conspiracy = pd.concat([df_conspiracy, data_fil])
df_conspiracy = df_conspiracy.drop(columns=["labels"])
print(df_conspiracy.head())
print(df_conspiracy["conspiracy"].value_counts())
df_conspiracy.to_csv("data/df_conspiracy.csv", index=False)

df_political = data.copy()
df_political["political"] = df_political["labels"].apply(
    lambda x: 1 if "political" in x else 0
)
data_fil = df_political[df_political["political"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(3 * factor)]
df_political = pd.concat([df_political, data_fil])
df_political = df_political.drop(columns=["labels"])
print(df_political.head())
print(df_political["political"].value_counts())
df_political.to_csv("data/df_political.csv", index=False)

df_country = data.copy()
df_country["country"] = df_country["labels"].apply(lambda x: 1 if "country" in x else 0)
data_fil = df_country[df_country["country"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(5 * factor)]
df_country = pd.concat([df_country, data_fil])
df_country = df_country.drop(columns=["labels"])
print(df_country.head())
print(df_country["country"].value_counts())
df_country.to_csv("data/df_country.csv", index=False)

df_rushed = data.copy()
df_rushed["rushed"] = df_rushed["labels"].apply(lambda x: 1 if "rushed" in x else 0)
data_fil = df_rushed[df_rushed["rushed"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(factor)]
df_rushed = pd.concat([df_rushed, data_fil])
df_rushed = df_rushed.drop(columns=["labels"])
print(df_rushed.head())
print(df_rushed["rushed"].value_counts())
df_rushed.to_csv("data/df_rushed.csv", index=False)

df_ingredients = data.copy()
df_ingredients["ingredients"] = df_ingredients["labels"].apply(
    lambda x: 1 if "ingredients" in x else 0
)
data_fil = df_ingredients[df_ingredients["ingredients"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(3 * factor)]
df_ingredients = pd.concat([df_ingredients, data_fil])
df_ingredients = df_ingredients.drop(columns=["labels"])
print(df_ingredients.head())
print(df_ingredients["ingredients"].value_counts())
df_ingredients.to_csv("data/df_ingredients.csv", index=False)

df_side_effect = data.copy()
df_side_effect["side-effect"] = df_side_effect["labels"].apply(
    lambda x: 1 if "side-effect" in x else 0
)
data_fil = df_side_effect[df_side_effect["side-effect"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(factor / 3)]
df_side_effect = pd.concat([df_side_effect, data_fil])
df_side_effect = df_side_effect.drop(columns=["labels"])
print(df_side_effect.head())
print(df_side_effect["side-effect"].value_counts())
df_side_effect.to_csv("data/df_side_effect.csv", index=False)

df_ineffective = data.copy()
df_ineffective["ineffective"] = df_ineffective["labels"].apply(
    lambda x: 1 if "ineffective" in x else 0
)
data_fil = df_ineffective[df_ineffective["ineffective"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(factor)]
df_ineffective = pd.concat([df_ineffective, data_fil])
df_ineffective = df_ineffective.drop(columns=["labels"])
print(df_ineffective.head())
print(df_ineffective["ineffective"].value_counts())
df_ineffective.to_csv("data/df_ineffective.csv", index=False)

df_religious = data.copy()
df_religious["religious"] = df_religious["labels"].apply(
    lambda x: 1 if "religious" in x else 0
)
data_fil = df_religious[df_religious["religious"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(factor * 6)]
df_religious = pd.concat([df_religious, data_fil])
df_religious = df_religious.drop(columns=["labels"])
print(df_religious.head())
print(df_religious["religious"].value_counts())
df_religious.to_csv("data/df_religious.csv", index=False)

df_none = data.copy()
df_none["none"] = df_none["labels"].apply(lambda x: 1 if "none" in x else 0)
data_fil = df_none[df_none["none"] == 1]
data_fil = data_fil.loc[data_fil.index.repeat(3 * factor)]
df_none = pd.concat([df_none, data_fil])
df_none = df_none.drop(columns=["labels"])
print(df_none.head())
print(df_none["none"].value_counts())
df_none.to_csv("data/df_none.csv", index=False)

# delete all files in the folder "/model"
folder = "models"
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print("Failed to delete %s. Reason: %s" % (file_path, e))