import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from ai_model.resources import get_learning_resources_for_missing_and_improving_skills
from ai_model.resources import get_learning_resources
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import os

# Load the dataset containing skill information and skill assessments
df = pd.read_csv(r'C:\Users\BRITNEY\Desktop\skill_assessment\dataset\skills_dataset.csv', on_bad_lines='warn')

# Clean up column names to remove any extra spaces
df.columns = df.columns.str.strip()

# Assuming your DataFrame is called df


# Initialize LabelEncoder for skill_name and skill_type
label_encoder_skill_name = LabelEncoder()
label_encoder_skill_type = LabelEncoder()

# Fit and encode the 'skill_name' column
df['skill_name_encoded'] = label_encoder_skill_name.fit_transform(df['skill_name'])

# Check if skill_type exists in the dataset and encode it
if 'skill_type' in df.columns:
    df['skill_type_encoded'] = label_encoder_skill_type.fit_transform(df['skill_type'])
else:
   
    df['skill_type_encoded'] = np.random.randint(0, 5, size=len(df))  # Randomly assign skill types

# Add placeholder 'score' and 'skill_gap' columns (replace with actual data in production)
df['score'] = np.random.randint(1, 6, size=len(df))  # Random scores between 1 and 5
df['skill_gap'] = np.random.randint(0, 2, size=len(df))  # Random binary gap indicator (0 = no gap, 1 = gap)

# Define input (X) and output (y) for the model
X = df[['skill_name_encoded', 'skill_type_encoded', 'score']]  # Features (input)
y = df['skill_gap']  # Target (output)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define transformers for each type of data
categorical_columns = ['skill_name_encoded']
numerical_columns = ['score']

categorical_transformer = Pipeline(steps=[ 
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

numerical_transformer = Pipeline(steps=[ 
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler())  # Scaling the numeric features
])

# Combine both transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[ 
        ('cat', categorical_transformer, categorical_columns), 
        ('num', numerical_transformer, numerical_columns), 
        ('skill_type', 'passthrough', ['skill_type_encoded'])  # Use encoded version of skill_type 
    ])
# Oversample the minority class using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Create a pipeline with preprocessing and classification
model_pipeline = Pipeline(steps=[ 
    ('preprocessor', preprocessor), 
    ('classifier', GradientBoostingClassifier(random_state=42)) 
])

# Hyperparameter tuning with RandomizedSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 500],
    'classifier__max_depth': [3, 5, 10],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__subsample': [0.8, 0.9, 1.0]
}

randomized_search = RandomizedSearchCV(
    estimator=model_pipeline, 
    param_distributions=param_grid, 
    n_iter=10, 
    cv=3, 
    verbose=2, 
    n_jobs=-1
)
randomized_search.fit(X_train_res, y_train_res)

# Evaluate on the test set
best_model = randomized_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Accuracy and classification report
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Save the trained model for future use
joblib.dump(best_model, 'best_skill_gap_model.pkl')


# Save the encoders for skill_name and skill_type
encoders_dir = os.path.join('skill_assessment', 'skill_assessment', 'encoders')

# Create the directory if it doesn't exist
if not os.path.exists(encoders_dir):
    os.makedirs(encoders_dir)
   

# Save the encoder files
joblib.dump(label_encoder_skill_name, os.path.join(encoders_dir, 'label_encoder_skill_name.pkl'))
joblib.dump(label_encoder_skill_type, os.path.join(encoders_dir, 'label_encoder_skill_type.pkl'))

import pandas as pd

# Skill type mapping
skill_type_mapping = {
    0: 'Technical Skills',
    1: 'Soft Skills',
    2: 'Management Skills',
    3: 'Analytical',
    4: 'Creative',
    # Add more mappings as needed
}

# Calculate the percentage score for each skill type based on user's assessment results
def calculate_percentage_score(df, skill_type_col='skill_type_encoded', score_col='score'):

    total_scores = df.groupby(skill_type_col)[score_col].max().sum()

    # Check if total_scores is zero to avoid division by zero
    if total_scores == 0:
        print("Warning: Total scores are zero, which means no valid scores for any skill type.")
        return pd.Series()  # Return empty series in case of invalid total score

    # Calculate the total score achieved by the user for each skill type
    user_scores = df.groupby(skill_type_col)[score_col].sum()

    # Normalize the highest skill type score to 100%
    max_user_score = user_scores.max()
    if max_user_score > 0:  # Avoid division by zero if max_user_score is 0
        user_scores = (user_scores / max_user_score) * 100

    # Calculate the percentage score for each skill type
    percentage_scores = user_scores.apply(lambda x: f"{x:.1f}%")  # Format with one decimal place

    # Map the encoded skill types to their actual names
    percentage_scores.index = percentage_scores.index.map(skill_type_mapping)

    # Filter out any '0.00%' or invalid data
    percentage_scores = percentage_scores[percentage_scores != '0.0%']

    return percentage_scores


# Call the function to calculate the percentage score
percentage_scores = calculate_percentage_score(df)

# Print out the percentage scores for each skill type
print("\nPercentage Scores by Skill Type:")
for skill_type, score in percentage_scores.items():
    print(f" {skill_type}: {score}")



# Function to categorize skills into strong, to improve, and missing
def categorize_skills(df, score_col='score'):
    # Categorize skills based on score
    strong_skills = df[df[score_col] >= 4]  # Strong skills where score >= 4
    skills_to_improve = df[(df[score_col] >= 2) & (df[score_col] < 4)]  # Skills to improve with score between 2 and 3
    missing_skills = df[df[score_col] <= 1]  # Missing skills with score <= 1
    
    # Get learning recommendations for each category
    skills_to_improve_with_recommendations = get_learning_resources(skills_to_improve['skill_name'].tolist())
    missing_skills_with_recommendations = get_learning_resources(missing_skills['skill_name'].tolist())
    
    return strong_skills[['skill_name']], skills_to_improve_with_recommendations, missing_skills_with_recommendations

# Call the function to categorize skills and get recommendations
strong_skills, skills_to_improve_with_recommendations, missing_skills_with_recommendations = categorize_skills(df)

print("\nStrong Skills:")
print("\n".join(strong_skills['skill_name'].values))


# Function to print formatted recommendations
def print_recommendations(skills_with_recommendations, title):
    print(f"\n{title}:")
    for skill, recommendation in skills_with_recommendations.items():
        if isinstance(recommendation, list):
            # If there's no recommendation, print a message
            print(f"- {skill}: No resources available for this skill.")
        elif isinstance(recommendation, dict):
            # If there's a course available, format it properly
            course = recommendation.get('course', 'No course available')
            link = recommendation.get('link', '')
            print(f"- {skill}: {course} - {link}")
        print("-" * 60)  # Separator line for clarity

# Sample usage with 'skills_to_improve_with_recommendations' and 'missing_skills_with_recommendations'
print_recommendations(skills_to_improve_with_recommendations, "Skills to Improve with Recommendations")
print_recommendations(missing_skills_with_recommendations, "Missing Skills with Recommendations")


# Function to get learning recommendations for missing or to improve skills
def get_learning_recommendations(skills, skills_to_improve):
    missing_skills = [
        'Data Analysis', 'Music Performance and Production', 'Health Informatics', 'Healthcare Management'
    ]
    
    # Filter the skills dataframe based on the provided lists
    skills_to_improve_df = skills[skills['skill_name'].isin(skills_to_improve)]  # Filtering based on user input skills
    missing_skills_df = skills[skills['skill_name'].isin(missing_skills)]  # Missing skill names predefined
    
    # Apply the learning resources logic for each skill
    skills['learning_resource'] = skills['skill_name'].apply(
        lambda skill_name: get_learning_resources_for_missing_and_improving_skills(skill_name, skills_to_improve_df)
    )
    
    return skills[['skill_name', 'learning_resource']]

# Function to categorize skills into strong, to improve, and missing
def categorize_skills_and_recommendations(df, score_col='score'):
    strong_skills = df[df[score_col] >= 4]  # Strong skills where score >= 4
    skills_to_improve = df[(df[score_col] >= 2) & (df[score_col] < 4)]  # Skills to improve with score between 2 and 3
    missing_skills = df[df[score_col] <= 1]  # Missing skills with score <= 1

    # Get learning recommendations for each category by passing both the full skills dataframe and filtered data
    skills_to_improve_with_recommendations = get_learning_recommendations(df, skills_to_improve['skill_name'])
    missing_skills_with_recommendations = get_learning_recommendations(df, missing_skills['skill_name'])

    return skills_to_improve_with_recommendations, missing_skills_with_recommendations

# Call the function to categorize skills and get recommendations
skills_to_improve_with_recommendations, missing_skills_with_recommendations = categorize_skills_and_recommendations(df)



