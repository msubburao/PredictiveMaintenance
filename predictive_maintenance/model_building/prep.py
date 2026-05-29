# # for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/msubburao/predictivemaintenanceds/engine_data.csv"
engds_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Shape of the dataset is ", engds_df.shape)

# Create a copy for cleaning
clean_df = engds_df.copy()

# Standardize column names for easier coding
clean_df.columns = (
    clean_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Updated dataset info ", clean_df.info())

# Define the target variable for the classification task
target = 'engine_condition'

# List of numerical features in the dataset
numeric_features = [
    'engine_rpm',     # Engine RPM
    'lub_oil_pressure', # Lube Oil Pressure
    'fuel_pressure', # Fuel Pressure
    'coolant_pressure', # Coolant Pressure
    'lub_oil_temp', # Lube Oil Temperature
    'coolant_temp' # Coolant Temperature
]


# Define predictor matrix (X) using selected numeric and categorical features
X = clean_df.drop(target, axis=1)

# Define target variable
y = clean_df[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

# Store Train and Test Datasets in Local

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]


# Upload train and test datasets into huggingface back

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="msubburao/predictivemaintenanceds",
        repo_type="dataset",
    )
