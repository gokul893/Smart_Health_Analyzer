import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import os

warnings.filterwarnings('ignore')

try:
    # --- Step 1: Load & Combine Datasets ---
    print("Loading thyroid disease datasets...")

    file_paths = [
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\hypothyroid.csv',
        'C:\\Users\\ngoku\\OneDrive\\Desktop\\Mini_Project1\\Dataset\\cleaned_dataset_Thyroid1.csv'
    ]

    # Load each dataset and drop within-file duplicates
    df_list = [pd.read_csv(fp).drop_duplicates().reset_index(drop=True) for fp in file_paths]
    df_combined = pd.concat(df_list, ignore_index=True)

    # Remove cross-file duplicates using a row hash
    df_combined['row_hash'] = df_combined.apply(lambda r: hash(tuple(r)), axis=1)
    df_combined.drop_duplicates(subset=['row_hash'], inplace=True)
    df_combined.drop(columns=['row_hash'], inplace=True)

    print(f"Combined dataset size after duplicate removal: {df_combined.shape[0]} samples.")

    # --- Step 2: Cleaning & Standardization ---
    df_combined.replace('?', np.nan, inplace=True)

    # Detect and rename target column
    target_candidates = ['target', 'binaryClass', 'Class']
    target_col = [col for col in target_candidates if col in df_combined.columns][0]
    df_combined.rename(columns={target_col: 'target'}, inplace=True)

    # Encode target: 0 = negative, 1 = positive
    negative_values = ['negative', 'P', 0, '0', '0.0']
    df_combined['target'] = df_combined['target'].apply(lambda x: 0 if str(x).strip() in negative_values else 1)

    # Drop irrelevant columns if present
    df_combined.drop(['referral_source', 'patient_id', 'TBG'], axis=1, inplace=True, errors='ignore')

    numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    categorical_cols = [col for col in df_combined.columns if col not in numeric_cols and col != 'target']

    # Clean numeric columns
    for col in numeric_cols:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
            df_combined[col].fillna(df_combined[col].median(), inplace=True)

    # Encode categorical columns
    encoders = {}
    for col in categorical_cols:
        df_combined[col] = df_combined[col].astype(str)
        mode_val = df_combined[col].mode()[0]
        df_combined[col].replace('nan', mode_val, inplace=True)
        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col])
        encoders[col] = le

    df_combined.dropna(subset=['target'], inplace=True)
    df_combined['target'] = df_combined['target'].astype(int)

    print("\nClass balance after cleaning:")
    print(df_combined['target'].value_counts())

    # --- Step 3: Split Data ---
    X = df_combined.drop('target', axis=1)
    y = df_combined['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Step 4: Address Class Imbalance with SMOTE (on training data) ---
    print("\nApplying SMOTE to the training data...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Training data size after SMOTE: {X_train_smote.shape[0]} samples.")
    print("\nClass balance after SMOTE:")
    print(y_train_smote.value_counts())

    # --- Step 5: Train Model on SMOTE-balanced data ---
    print("\nTraining the Gradient Boosting model on SMOTE-balanced data...")
    thyroid_model_smote = GradientBoostingClassifier(

        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    )
    thyroid_model_smote.fit(X_train_smote, y_train_smote)
    print("Model training complete.")

    # --- Step 6: Evaluate Model Performance on original test set ---
    print("\nEvaluating the retrained model on the original test set...")
    y_pred_smote = thyroid_model_smote.predict(X_test)
    accuracy_smote = accuracy_score(y_test, y_pred_smote)
    precision_smote = precision_score(y_test, y_pred_smote)
    recall_smote = recall_score(y_test, y_pred_smote)
    f1_smote = f1_score(y_test, y_pred_smote)

    print("\n" + "="*50)
    print("Hold-out Test Performance (SMOTE, Retrained Model):")
    print("="*50)
    print(f"Accuracy : {accuracy_smote * 100:.2f}%")
    print(f"Precision: {precision_smote * 100:.2f}%")
    print(f"Recall   : {recall_smote * 100:.2f}%")
    print(f"F1-Score : {f1_smote * 100:.2f}%")

    # --- Step 7: 5-Fold Cross-Validation with SMOTE in each fold ---
    print("\nPerforming 5-Fold Cross-Validation with SMOTE...")
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', GradientBoostingClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3
        ))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_smote = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')

    print("\nCross-Validation F1 Scores (with SMOTE):", np.round(cv_scores_smote, 4))
    print("Mean F1 Score (5-fold with SMOTE):", np.round(cv_scores_smote.mean(), 4))

    # --- Step 8: Save the Final Model and Preprocessing Data ---
    print("\nSaving the final thyroid model and preprocessing data...")

    preprocessing_data = {
        'encoders': encoders,
        'training_data_medians': df_combined[numeric_cols].median().to_dict(),
        'column_order': X.columns.tolist()
    }
    
    # Define the correct save path inside the 'Models' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, 'thyroid_model.pkl')
    preprocessor_save_path = os.path.join(script_dir, 'thyroid_preprocessing_data.pkl')

    # Save the files to the correct paths
    joblib.dump(thyroid_model_smote, model_save_path)
    joblib.dump(preprocessing_data, preprocessor_save_path)
    
    print(f"Successfully saved model to: {model_save_path}")
    print(f"Successfully saved preprocessing data to: {preprocessor_save_path}")

except FileNotFoundError as e:
    print("\nError: Required thyroid dataset not found.")
    print(f"Missing file: {e.filename}")
except Exception as e:
    print("\nAn unexpected error occurred:", e)