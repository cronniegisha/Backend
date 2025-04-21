import pandas as pd
import joblib
import os
import django
import sys

from django.conf import settings

# Set up Django
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

# === Setup paths ===
BASE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
ENCODER_DIR = os.path.join(BASE_DIR, 'skill_assessment', 'encoders')

skill_name_encoder_path = os.path.join(ENCODER_DIR, 'label_encoder_skill_name.pkl')
skill_type_encoder_path = os.path.join(ENCODER_DIR, 'label_encoder_skill_type.pkl')
model_path = os.path.join(BASE_DIR, 'skill_gap_predictor_model.pkl')

# === Load Encoders and Model ===
print("🔍 Loading encoders from:")
print("   Skill Name Encoder:", skill_name_encoder_path)
print("   Skill Type Encoder:", skill_type_encoder_path)

label_encoder_skill_name = joblib.load(skill_name_encoder_path)
label_encoder_skill_type = joblib.load(skill_type_encoder_path)

print("✅ Successfully loaded the skill_name and skill_type encoders.")

pipeline = joblib.load(model_path)

# === Main Function ===
def identify_skill_gaps(user_skills):
    """
    user_skills: list of dicts with keys 'skill_name', 'skill_type', and 'score'
    Example:
    [
        {'skill_name': 'Python Programming', 'skill_type': 'Technical', 'score': 2},
        {'skill_name': 'Communication', 'skill_type': 'Soft', 'score': 4},
    ]
    """
    
    df = pd.DataFrame(user_skills, columns=["skill_name", "skill_type"])


    # Encode skill names and types
    df['skill_name_encoded'] = label_encoder_skill_name.transform(df['skill_name'])
    df['skill_type_encoded'] = label_encoder_skill_type.transform(df['skill_type'])
    # Add the 'score' column. If it's calculated based on other columns, ensure it is defined.
    # For example:
    df['score'] = df['skill_name_encoded'] * 0.5 + df['skill_type_encoded'] * 0.5  # Example score calculation

    features = df[['skill_name_encoded', 'skill_type_encoded', 'score']]

    # Define the logic for identifying missing skills (e.g., low score or skills that aren't in the required set)
    required_skills = ["Python Programming", "Git Version Control", "Marine Engineering", "Aerospace Engineering", "SQL Databases","Software Testing"]
    missing_skills = [skill for skill in required_skills if skill not in df['skill_name'].values]
    
    # Return skill analysis and missing skills
    skill_analysis = df[['skill_name', 'score']]
    
    predictions = pipeline.predict(features)

    # Categorize each skill
    categorized_skills = []
    for idx, row in df.iterrows():
        skill_name = row['skill_name']
        score = row['score']
        gap_prediction = predictions[idx]

        if gap_prediction == 1:
            status = f"Missing skill: {skill_name}"
        elif gap_prediction == 0 and score >= 3:
            status = f"Strong skill: {skill_name}"
        else:
            status = f"Skill to improve: {skill_name}"

        categorized_skills.append(status)

    return categorized_skills

# === Debug Script ===
if __name__ == "__main__":
    test_input = [
        {'skill_name': 'Python Programming', 'skill_type': 'Technical', 'score': 2}
    ]
    result = identify_skill_gaps(test_input)
    for line in result:
        print(line)
