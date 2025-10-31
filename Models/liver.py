import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

try:
    # --- Step 1: Load and Standardize All Suitable Datasets Individually ---
    print("Loading all suitable liver disease datasets...")

    file_paths = [
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\Liver Patient Dataset (LPD)_train.csv',
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\indian_liver_patient (1).csv',
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\indian_liver_patient (2).csv',
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\Indian Liver Patient Dataset (ILPD).csv'
    ]

    column_mapping = {
        'Age of the patient': 'Age', 'Gender of the patient': 'Gender',
        'Total Bilirubin': 'Total_Bilirubin', 'Direct Bilirubin': 'Direct_Bilirubin',
        'Alkphos Alkaline Phosphotase': 'Alkaline_Phosphotase', 'Sgpt Alamine Aminotransferase': 'Alamine_Aminotransferase',
        'Sgot Aspartate Aminotransferase': 'Aspartate_Aminotransferase', 'Total Protiens': 'Total_Protiens',
        'ALB Albumin': 'Albumin', 'A/G Ratio Albumin and Globulin Ratio': 'Albumin_and_Globulin_Ratio',
        'Result': 'Dataset', 'Selector': 'Dataset', 'TB': 'Total_Bilirubin', 'DB': 'Direct_Bilirubin',
        'Alkphos': 'Alkaline_Phosphotase', 'Sgpt': 'Alamine_Aminotransferase', 'Sgot': 'Aspartate_Aminotransferase',
        'TP': 'Total_Protiens', 'ALB': 'Albumin', 'A/G Ratio': 'Albumin_and_Globulin_Ratio'
    }

    final_cols = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                  'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                  'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']

    clean_df_list = []
    for fp in file_paths:
        df = pd.read_csv(fp, encoding='latin1')
        df.columns = df.columns.str.strip()
        df.rename(columns=column_mapping, inplace=True)
        df = df.reindex(columns=final_cols)
        clean_df_list.append(df)

    df_combined = pd.concat(clean_df_list, ignore_index=True)
    df_combined.drop_duplicates(inplace=True)

    print(f"Successfully loaded and combined {len(clean_df_list)} datasets.")

    # --- Step 2: Preprocessing ---
    df_combined['Gender'] = df_combined['Gender'].replace({'Male': 1, 'Female': 0, 'female': 0})
    df_combined['Gender'] = pd.to_numeric(df_combined['Gender'], errors='coerce')
    df_combined['Gender'].fillna(df_combined['Gender'].mode()[0], inplace=True)

    df_combined['Dataset'] = df_combined['Dataset'].replace(2, 0)
    df_combined['Albumin_and_Globulin_Ratio'].fillna(df_combined['Albumin_and_Globulin_Ratio'].median(), inplace=True)

    df_combined.dropna(inplace=True)
    print(f"Final dataset size after cleaning and imputing: {df_combined.shape[0]} samples.")

    # --- Step 3: Prepare Data for Cross-Validation ---
    X = df_combined.drop('Dataset', axis=1)
    y = df_combined['Dataset']
    training_columns = X.columns.tolist()

    # --- Step 4: Use 5-Fold Cross-Validation for Robust Evaluation ---
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

    # --- Step 5: Display Average Performance ---
    print("\n" + "="*50)
    print("Cross-Validated Liver Disease Model Performance:")
    print("="*50)
    print(f"Average Accuracy: {np.mean(accuracies) * 100:.2f}%")
    print(f"Average Precision: {np.mean(precisions) * 100:.2f}%")
    print(f"Average Recall: {np.mean(recalls) * 100:.2f}%")
    print(f"Average F1-Score: {np.mean(f1s) * 100:.2f}%")
    print("="*50)
    print("These scores provide a more realistic estimate of the model's performance.")

    # --- Step 6: Train and Save the Final Model and Preprocessing Data ---
    print("\nTraining final model on all available data...")
    final_liver_pipeline = pipeline.fit(X, y)
    print("Final model training complete.")

    preprocessing_data = {
        'imputation_values': {
            'Gender_mode': df_combined['Gender'].mode()[0],
            'AG_Ratio_median': df_combined['Albumin_and_Globulin_Ratio'].median()
        },
        'column_order': training_columns
    }

    # Define the correct save path inside the 'Models' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, 'liver_model.pkl')
    preprocessor_save_path = os.path.join(script_dir, 'liver_preprocessing_data.pkl')

    print("\nSaving model and preprocessing data to disk...")
    joblib.dump(final_liver_pipeline, model_save_path)
    joblib.dump(preprocessing_data, preprocessor_save_path)
    
    print(f"Successfully saved model to: {model_save_path}")
    print(f"Successfully saved preprocessing data to: {preprocessor_save_path}")

except FileNotFoundError as e:
    print(f"\nAn error occurred: A required liver disease dataset was not found.")
except Exception as e:
    print(f"\nAn error occurred during model building: {e}")