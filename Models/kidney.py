import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib # Import joblib for saving the model
import warnings
import os

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    # --- Step 1: Load and Standardize All Datasets ---
    print("Loading all real-world clinical datasets...")
    file_paths = [
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\kidney_disease (1) (1).csv',
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\kidney_disease (2) (1).csv',
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\kidney_disease (5).csv',
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\kidney_disease (6).csv',
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\kidney_disease_dataset (1).csv'
    ]

    # **Define the correct, final column names that will be enforced on every file.**
    standard_columns = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 
        'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 
        'appet', 'pe', 'ane', 'class'
    ]
    
    df_list = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        # Drop the 'id' column from the individual file if it exists
        df.drop('id', axis=1, inplace=True, errors='ignore')
        # **CRITICAL FIX: Enforce the standard column names BEFORE combining.**
        # This prevents the "Length mismatch" error.
        df.columns = standard_columns
        df_list.append(df)

    # Now, concatenation will align the data correctly
    df_combined = pd.concat(df_list, ignore_index=True)
    df_combined.drop_duplicates(inplace=True)
    print(f"Successfully loaded and combined {len(df_list)} datasets.")

    # --- Step 2: Intensive Data Cleaning ---
    print("Starting intensive data cleaning...")

    # Standardize missing/incorrect values
    df_combined.replace(['\t?', '?', '\t\tyes', '\tyes', '\tno', 'ckd\t', ' '],
                          [np.nan, np.nan, 'yes', 'yes', 'no', 'ckd', np.nan], inplace=True)

    # Clean and encode the target variable 'class'
    df_combined['class'] = df_combined['class'].str.strip().replace({
    # Categories for 1 (Disease or At-Risk)
        'ckd': 1,
        'Moderate_Risk': 1,
        'High_Risk': 1,
        'Severe_Disease': 1,

        # Categories for 0 (No Disease or Low-Risk)
        'notckd': 0,
        'No_Disease': 0,
        'Low_Risk': 0
        })

    # Define which columns are numeric and categorical
    numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    categorical_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    # Impute missing numeric values with the median
    for col in numeric_cols:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        df_combined[col].fillna(df_combined[col].median(), inplace=True)

    # Impute missing categorical values with the mode and then label encode them
    encoders = {}
    modes = {}
    for col in categorical_cols:
        df_combined[col] = df_combined[col].astype(str)
        mode_val = df_combined[col].mode()[0]
        modes[col] = mode_val
        df_combined[col].fillna(mode_val, inplace=True)

        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col])
        encoders[col] = le

    # Drop any rows where the target variable is still missing
    df_combined.dropna(subset=['class'], inplace=True)
    print(f"Dataset size after cleaning and imputing: {df_combined.shape[0]} samples.")

    # --- Step 3: Prepare Data for Modeling ---
    X = df_combined.drop('class', axis=1)
    y = df_combined['class'].astype(int)

    # --- Step 4: Define Pipeline and Use Cross-Validation for Robust Evaluation ---
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', GradientBoostingClassifier())
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1s = [], [], [], []

    print("\nRunning 5-fold cross-validation...")
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    print("\n" + "="*50)
    print("Cross-Validated Kidney Disease Model Performance:")
    print("="*50)
    print(f"Average Accuracy: {np.mean(accuracies) * 100:.2f}%")
    print(f"Average Precision: {np.mean(precisions) * 100:.2f}%")
    print(f"Average Recall: {np.mean(recalls) * 100:.2f}%")
    print(f"Average F1-Score: {np.mean(f1s) * 100:.2f}%")
    print("="*50)

    # --- Step 5: Train and Save the Final Model and Preprocessing Objects ---
    print("\nTraining final model on all available data...")
    pipeline.fit(X, y)
    print("Final model training complete.")

    # Create a dictionary to hold all necessary preprocessing objects
    preprocessing_data = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoders': encoders,
        'modes': modes,
        'training_data_medians': df_combined[numeric_cols].median().to_dict(), # Use cleaned medians
        'column_order': X.columns.tolist() # Save the column order
    }
    
    # --- Step 5: Train and Save the Final Model and Preprocessing Objects ---
    print("\nTraining final model on all available data...")
    pipeline.fit(X, y)
    print("Final model training complete.")

    # Create a dictionary to hold all necessary preprocessing objects
    preprocessing_data = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoders': encoders,
        'modes': modes,
        'training_data_medians': df_combined[numeric_cols].median().to_dict(),
        'column_order': X.columns.tolist()
    }

    # **NEW: Define the correct save path inside the 'Models' folder**
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, 'kidney_disease_model.pkl')
    preprocessor_save_path = os.path.join(script_dir, 'kidney_disease_preprocessing_data.pkl')
    
    # Save the trained model and the preprocessing data to the correct paths
    joblib.dump(pipeline, model_save_path)
    joblib.dump(preprocessing_data, preprocessor_save_path)
    
    print(f"\nSuccessfully saved model to: {model_save_path}")
    print(f"Successfully saved preprocessing data to: {preprocessor_save_path}")

except FileNotFoundError as e:
    print(f"\nAn error occurred: A required kidney disease dataset was not found.")
    print(e)
except Exception as e:
    print(f"\nAn error occurred during training: {e}")