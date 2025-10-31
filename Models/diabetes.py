import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

try:
    # --- Step 1: Load and Standardize Each of the Four Clinical Datasets ---

    # Dataset 1: Original
    df1 = pd.read_csv('C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\diabetes_prediction_dataset.csv')
    df1.rename(columns={'HbA1c_level': 'hba1c_level'}, inplace=True)
    df1 = df1[['age', 'bmi', 'blood_glucose_level', 'hba1c_level', 'diabetes']]

    # Dataset 2: PIMA Indians
    df2 = pd.read_csv('C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\diabetes (1).csv')
    df2.rename(columns={'Age': 'age', 'BMI': 'bmi', 'Glucose': 'blood_glucose_level', 'Outcome': 'diabetes'}, inplace=True)
    df2 = df2[['age', 'bmi', 'blood_glucose_level', 'diabetes']] # Note: No HbA1c in this dataset

    # Dataset 3: Bangladesh
    df3 = pd.read_csv('C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\Diabetes_Bangladesh.csv')
    df3.rename(columns={'glucose': 'blood_glucose_level', 'diabetic': 'diabetes'}, inplace=True)
    df3['diabetes'] = (df3['diabetes'] == 'Yes').astype(int)
    df3 = df3[['age', 'bmi', 'blood_glucose_level', 'diabetes']] # Note: No HbA1c

    # Dataset 4: "Dataset of Diabetes"
    df4 = pd.read_csv('C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\Dataset of Diabetes .csv')
    df4.rename(columns={'AGE': 'age', 'BMI': 'bmi', 'HbA1c': 'hba1c_level', 'CLASS': 'diabetes'}, inplace=True)
    df4['diabetes'] = (df4['diabetes'].str.strip() == 'Y').astype(int)
    df4 = df4[['age', 'bmi', 'hba1c_level', 'diabetes']] # Note: No direct Glucose reading

    print("Successfully loaded and standardized all four clinical datasets.")

    # --- Step 2: Combine and Impute ---
    df_combined = pd.concat([df1, df2, df3, df4], ignore_index=True)

    for col in ['blood_glucose_level', 'hba1c_level', 'bmi']:
        df_combined[col].replace(0, pd.NA, inplace=True)
        median_val = df_combined[col].median()
        df_combined[col].fillna(median_val, inplace=True)

    df_combined.dropna(inplace=True)
    print(f"Created a combined and imputed dataset with {df_combined.shape[0]} samples.")

    # --- Step 3: Train and Evaluate the Fully Generalized Model ---
    X = df_combined.drop('diabetes', axis=1)
    y = df_combined['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    final_generalized_model = GradientBoostingClassifier()
    final_generalized_model.fit(X_train_resampled, y_train_resampled)

    y_pred = final_generalized_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "="*50)
    print("Fully Generalized Model Performance (4 Datasets):")
    print("="*50)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

    # --- Step 4: Save the Final Model and Preprocessing Data ---
    print("\nSaving the generalized diabetes model and preprocessing data...")
    
    preprocessing_data = {
        'training_data_medians': {
            'bmi': df_combined['bmi'].median(),
            'blood_glucose_level': df_combined['blood_glucose_level'].median(),
            'hba1c_level': df_combined['hba1c_level'].median()
        },
        'column_order': X.columns.tolist()
    }

    # Define the correct save path inside the 'Models' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, 'diabetes_model.pkl')
    preprocessor_save_path = os.path.join(script_dir, 'diabetes_preprocessing_data.pkl')

    # Save the files to the correct paths
    joblib.dump(final_generalized_model, model_save_path)
    joblib.dump(preprocessing_data, preprocessor_save_path)
    
    print(f"Successfully saved model to: {model_save_path}")
    print(f"Successfully saved preprocessing data to: {preprocessor_save_path}")

except FileNotFoundError as e:
    print(f"\nError: A dataset file was not found. Please check the file paths.")
except Exception as e:
    print(f"\nAn error occurred: {e}")