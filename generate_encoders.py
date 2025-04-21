import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Absolute base directory (adjust this if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create encoders directory if it doesn't exist
encoders_dir = os.path.join(BASE_DIR, 'skill_assessment', 'encoders')
os.makedirs(encoders_dir, exist_ok=True)

# Dataset path
csv_path = os.path.join(BASE_DIR, 'dataset', 'skills_dataset.csv')

# Load dataset
try:
    df = pd.read_csv(r'C:\Users\BRITNEY\Desktop\skill_assessment\dataset\skills_dataset.csv', on_bad_lines='warn')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Dataset CSV not found at:", csv_path)
    exit()

# Skill name encoder
skill_name_file = os.path.join(encoders_dir, 'label_encoder_skill_name.pkl')
if 'skill_name' in df.columns:
    le_name = LabelEncoder()
    le_name.fit(df['skill_name'])
    joblib.dump(le_name, skill_name_file)
    print(f"✅ Skill name encoder saved to: {skill_name_file}")
else:
    print("❌ 'skill_name' column missing in dataset.")

# Skill type encoder
skill_type_file = os.path.join(encoders_dir, 'label_encoder_skill_type.pkl')
if 'skill_type' in df.columns:
    le_type = LabelEncoder()
    le_type.fit(df['skill_type'])
    joblib.dump(le_type, skill_type_file)
    print(f"✅ Skill type encoder saved to: {skill_type_file}")
else:
    print("❌ 'skill_type' column missing in dataset.")
