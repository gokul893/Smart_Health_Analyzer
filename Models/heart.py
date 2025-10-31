import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

try:
    # --- Step 1: Load and Standardize the Three Clinical Datasets
    df1 = pd.read_csv('C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\heart_cleveland_upload.csv')
    df1.rename(columns={'condition': 'target'}, inplace=True)

    df2 = pd.read_csv('C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\Heart_disease_cleveland_new.csv')

    df3 = pd.read_csv('C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\heart_disease_uci.csv')
    df3.rename(columns={'num': 'target'}, inplace=True)
    df3['target'] = (df3['target'] > 0).astype(int)

    print("Successfully loaded and standardized the three clinical datasets.")

    # --- Step 2: Combine the Datasets
    df_combined = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"Created a combined dataset with {df_combined.shape[0]} samples.")

    # --- Step 3: Robust Cleaning and Imputation
    # Drop non-predictive metadata columns before cleaning
    df_combined.drop(columns=['id', 'dataset'], inplace=True, errors='ignore')

    # Replace placeholder '?' with NaN for proper handling
    df_combined.replace('?', np.nan, inplace=True)

    # Identify numeric and categorical columns
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Impute numeric columns with the median
    for col in numeric_cols:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        median_val = df_combined[col].median()
        df_combined[col].fillna(median_val, inplace=True)

    # Impute categorical columns with the mode (most frequent value)
    for col in categorical_cols:
        mode_val = df_combined[col].mode()[0]
        df_combined[col].fillna(mode_val, inplace=True)

    # Drop any row that still contains any NaN value after imputation
    df_combined.dropna(inplace=True)
    print(f"Dataset size after robust imputing: {df_combined.shape[0]} samples.")

    # --- Step 4: Final Preprocessing
    # One-hot encode the clean categorical features
    df_final = pd.get_dummies(df_combined, columns=categorical_cols, drop_first=True)

    # --- Step 5: Train and Tune the Generalized Model
    X = df_final.drop('target', axis=1)
    y = df_final['target']

    # Ensure the target column is integer type
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # --- Step 6: Hyperparameter Tuning using GridSearchCV to improve the accuarcy
    print("\nStarting hyperparameter tuning to find the best model settings...")

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.8, 1.0]
    }

    # Set up the grid search with 3-fold cross-validation
    grid_search = GridSearchCV(estimator=GradientBoostingClassifier(),
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=3,
                               n_jobs=-1,
                               verbose=1)

    # Fit the grid search to find the best model
    grid_search.fit(X_train_resampled, y_train_resampled)

    print("\nHyperparameter tuning complete.")
    print(f"Best parameters found: {grid_search.best_params_}")

    # Use the best model found by the search
    best_heart_model = grid_search.best_estimator_


    # --- Step 7: Evaluate the Tuned Model
    y_pred = best_heart_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "="*50)
    print("Heart Disease Model Performance:")
    print("="*50)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

   # --- Step 8: Save the Tuned Model and Preprocessing Data ---
    print("\nSaving the tuned heart disease model and preprocessing data...")

    preprocessing_data = {
        'column_order': X.columns.tolist() 
    }

    # Define the correct save path inside the 'Models' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, 'heart_disease_model.pkl')
    preprocessor_save_path = os.path.join(script_dir, 'heart_disease_preprocessing_data.pkl')
    
    # Save the files to the correct paths
    joblib.dump(best_heart_model, model_save_path)
    joblib.dump(preprocessing_data, preprocessor_save_path)
    
    print(f"Successfully saved model to: {model_save_path}")
    print(f"Successfully saved preprocessing data to: {preprocessor_save_path}")

except FileNotFoundError as e:
    print(f"\nError: A dataset file was not found. Please ensure all heart disease CSV files are in the directory.")
except Exception as e:
    print(f"\nAn error occurred: {e}")