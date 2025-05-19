import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import joblib
import pickle
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder, StandardScaler

# File path - update this to your file path
user_data_path = "C:/Users/SHIRAH/Desktop/Test/career_matching/career_match/Cleaned_Users.csv"

# Function to clean skills column - improved to better handle different formats
def clean_skills_column(column):
    cleaned = []
    for entry in column:
        if isinstance(entry, str):
            # Handle both comma and space separated skills
            entry = entry.lower().strip()
            # First split by commas if they exist
            if ',' in entry:
                skills = [skill.strip() for skill in entry.split(',') if skill.strip()]
            else:
                # Otherwise split by spaces, but be careful with multi-word skills
                skills = [skill.strip() for skill in entry.split() if skill.strip()]
            cleaned.append(skills)
        elif isinstance(entry, list):
            cleaned.append([skill.strip().lower() for skill in entry if skill.strip()])
        else:
            cleaned.append([])
    return cleaned

def clean_interests_column(column):
    cleaned = []
    for entry in column:
        if isinstance(entry, str):
            # Handle both comma and space separated skills
            entry = entry.lower().strip()
            # First split by commas if they exist
            if ',' in entry:
                interests = [interest.strip() for interest in entry.split(',') if interest.strip()]
            else:
                # Otherwise split by spaces, but be careful with multi-word skills
                interests = [interest.strip() for interest in entry.split() if interest.strip()]
            cleaned.append(interests)
        elif isinstance(entry, list):
            cleaned.append([interest.strip().lower() for interest in entry if interest.strip()])
        else:
            cleaned.append([])
    return cleaned
def clean_education_column(df,column='education'):
    mapping = {
        'phd': 'phd',
        'doctorate': 'phd',
        'doctorate(phd/md)': 'phd',
        "bachelor's": "bachelor's",
        "bachelor'sdegree": "bachelor's",
        "bachelors": "bachelor's",
        "bachelorsdegree": "bachelor's",
        "master's": "master's",
        "master'sdegree": "master's",
        "masters": "master's",
        "mastersdegree": "master's",
        "highschooldiploma": "uacecertificate",
        "uacecertificate": "uacecertificate",
        "ucecertificate": "ucecertificate"
    }

    def clean_level(level):
        if pd.isnull(level):
            return None
        key = str(level).strip().lower().replace(" ", "")
        return mapping.get(key)

    df[column] = df[column].apply(clean_level)
    df.dropna(subset=[column], inplace=True)
    return df


# Function to create a mapping between career names and their encoded values
def create_career_mapping(encoder, career_names):
    mapping = {}
    for i, name in enumerate(career_names):
        try:
            encoded = encoder.transform([name])[0]
            mapping[name] = encoded
            mapping[encoded] = name
        except:
            print(f"Warning: Could not encode career name: {name}")
    return mapping


# Main function to process data and train model
def train_career_model(user_data_path):
    print("Loading data...")
    user_data = pd.read_csv(user_data_path, encoding='latin1')
    user_data.columns = user_data.columns.str.strip().str.lower()
    print("Available columns:", user_data.columns.tolist())
    print(f"Loaded {len(user_data)} users.")
   

    # Clean column names
    user_data.columns = user_data.columns.str.strip().str.lower()
    
    print("Preprocessing data...")
    # Handle missing values
    user_data['skills'] = user_data['skills'].fillna('')
    user_data['interests'] = user_data['interests'].fillna('')
    user_data['age'] = pd.to_numeric(user_data['age'], errors='coerce').fillna(0)
    user_data['education'] = user_data['education'].fillna('')
    user_data['education'] = user_data['education'].astype(str)
    user_data['recommended_career'] = user_data['recommended_career'].fillna('').astype(str)

    
    
    # Clean and process skills
    print("Processing skills data...")
    user_data['skills'] = clean_skills_column(user_data['skills'])
    user_data['interests'] = clean_interests_column(user_data['interests'])
    user_data = clean_education_column(user_data, column='education')
    
        # 3. Drop rows with any missing or invalid key fields (after cleaning)
    required_columns = ['recommended_career', 'skills', 'education', 'age', 'interests']
    user_data = user_data.dropna(subset=required_columns)

        # 4. Reset index after cleaning to avoid index misalignment
    user_data = user_data.reset_index(drop=True)
    
    # Create a global skill vocabulary
    all_user_skills = set(skill for sublist in user_data['skills'] for skill in sublist)
    all_user_interests = set(interest for sublist in user_data['interests'] for interest in sublist)
    print(f"Found {len(all_user_skills)} unique skills across all data.")
    print(f"Found {len(all_user_interests)} unique interests across all data.")
    
        # Skills encoder
    skills_mlb = MultiLabelBinarizer()
    skills_mlb.fit(user_data['skills'])

    # Interests encoder
    interests_mlb = MultiLabelBinarizer()
    interests_mlb.fit(user_data['interests'])

    # Transform skills and interests
    user_skills_encoded = pd.DataFrame(
        skills_mlb.transform(user_data['skills']),
        columns=skills_mlb.classes_
    )

    user_interests_encoded = pd.DataFrame(
        interests_mlb.transform(user_data['interests']),
        columns=interests_mlb.classes_
    )

    
    # Create encoders for categorical variables
    print("Encoding categorical variables...")
    # Education/qualifications encoder
    education_encoder = LabelEncoder()
    education_values = list(set(
        user_data['education'].tolist()
       
    ))
    education_encoder.fit(education_values)
   

    # Encode categorical variables
    user_data['education_encoded'] = education_encoder.transform(user_data['education'])

    recommended_career_encoder = LabelEncoder()
    recommended_career_encoder.fit(user_data['recommended_career'])
    user_data['recommended_career_encoded'] = recommended_career_encoder.transform(user_data['recommended_career'])
    user_data['recommended_career'] = user_data['recommended_career'].str.lower().str.strip()

    
    # Scale numerical features
    scaler = StandardScaler()
    user_data['age_scaled'] = scaler.fit_transform(user_data[['age']])
    
    # Create feature matrices
    print("Creating feature matrices...")
    X_user = pd.concat((
        user_data[['education_encoded']],
        user_interests_encoded,
        user_skills_encoded
    ), axis=1)
    X_user.fillna(0, inplace=True)
    
    if 'recommended_career' in user_data.columns:
        y_user = user_data['recommended_career_encoded']
    else:
        print("Warning: No 'recommended_career' column found. Cannot train supervised model.")
        return None

    # Fix mixed-type column names
    X_user.columns = X_user.columns.astype(str)
    y_user = y_user.ravel()  # This flattens y into a 1D array, avoiding the column-vector issue

    
    # Handle class imbalance
    print("Handling class imbalance...")
    counter = Counter(y_user)
    print(f"Class distribution before resampling: {counter}")

    print(f"Length of X_user: {len(X_user)}")
    print(f"Length of y_user: {len(y_user)}")

    print("X_user head:")
    print(X_user.head())

    print("y_user head:")



    X_user = X_user.reset_index(drop=True)

        # Sanity check
    assert len(X_user) == len(y_user)
    
    
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_user, y_user)

    
    print(f"Class distribution after resampling: {Counter(y_resampled)}")
    
    # Split data for training
    print("Splitting data for training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    feature_names = X_train.columns.tolist()
    # Train model with hyperparameter tuning
    print("Training model with hyperparameter tuning...")
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced']
}

    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_iter=30,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = best_rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

    
    print("Cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
    
    # Save model and encoders
    print("Saving model and encoders...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_rf, "model/rf_model.pkl")
    joblib.dump(education_encoder, "model/education_encoder.pkl")
    joblib.dump(interests_mlb, "model/interests_encoder.pkl")
    joblib.dump(skills_mlb, "model/skills_encoder.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(recommended_career_encoder, "model/target_encoder.pkl")
    joblib.dump(feature_names, "model/feature_names.pkl")
    

    print("\nTraining complete!")
    
if __name__ == "__main__":
    train_career_model(user_data_path)
