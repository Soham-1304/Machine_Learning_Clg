"""
Exploratory Data Analysis — Netflix Movies and TV Shows Dataset
Assignment 2 (CO1)
"""

import pandas as pd

# ---------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------
df = pd.read_csv("netflix_titles.csv")

# ---------------------------------------------------------
# Step 2: Shape and columns
# ---------------------------------------------------------
print("Shape (rows, columns):", df.shape)
print("Columns:", df.columns.tolist())

# ---------------------------------------------------------
# Step 3: Data types and non-null counts
# ---------------------------------------------------------
df.info()

# Preview first few rows
print(df.head())

# ---------------------------------------------------------
# Step 4: Missing values
# ---------------------------------------------------------
missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing_counts, "missing_pct": missing_pct})
print(missing_summary[missing_summary["missing_count"] > 0])

# ---------------------------------------------------------
# Step 5: Duplicate rows
# ---------------------------------------------------------
print("Fully duplicated rows:", df.duplicated().sum())
print("Duplicate titles:", df['title'].duplicated().sum())

# ---------------------------------------------------------
# Step 6: Basic statistics (numeric columns)
# ---------------------------------------------------------
print(df.describe())

# Categorical spread
print(df['type'].value_counts())
print(df['rating'].value_counts())
print(df['country'].value_counts().head(10))

# ---------------------------------------------------------
# Step 7: Outlier check on release_year
# ---------------------------------------------------------
print(df['release_year'].describe())
old_titles = df[df['release_year'] < 1960]
print("Titles released before 1960:", len(old_titles))
print(old_titles[['title', 'release_year']])

# ---------------------------------------------------------
# Step 8: ML problem type discussion (see written report)
# ---------------------------------------------------------
# 1. Unsupervised clustering — group titles by genre/description similarity
#    (no existing labels for "type of show" beyond raw genre tags)
# 2. Supervised classification — predict `rating` (or `type`) from
#    `description`, `listed_in`, `duration` (rating is an existing label
#    on every row, making this a valid supervised setup)
