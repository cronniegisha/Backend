import joblib
import numpy as np
import pandas as pd
import os

def load_models(model_dir='model'):
    """Load all the trained models and encoders"""
    model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    skills_encoder = joblib.load(os.path.join(model_dir, "skills_encoder.pkl"))
    interests_encoder = joblib.load(os.path.join(model_dir, "interests_encoder.pkl"))
    education_encoder = joblib.load(os.path.join(model_dir, "education_encoder.pkl"))
    target_encoder = joblib.load(os.path.join(model_dir, "target_encoder.pkl"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))

    return model, skills_encoder, interests_encoder, education_encoder, target_encoder, feature_names


def predict_career(age, education, skills, interests, top_n=3):
    """Predict top N career recommendations based on user input"""
    try:
        # Load models and encoders
        model, skills_encoder, interests_encoder, education_encoder, target_encoder, feature_names = load_models()

        # Debug: print available career classes
        print(f"Available careers in model: {len(target_encoder.classes_)}")
        print(f"First few careers: {target_encoder.classes_[:5]}")

        # Preprocess input
        skills = [skill.strip().lower() for skill in skills]
        interests = [interest.strip().lower() for interest in interests]

        # Encode skills
        skills_df = pd.DataFrame(columns=skills_encoder.classes_)
        skills_df.loc[0] = 0
        for skill in skills:
            if skill in skills_encoder.classes_:
                skills_df.loc[0, skill] = 1
            else:
                print(f"⚠️ Unknown skill: {skill}")  # Handle unknown skills

        # Encode interests
        interests_df = pd.DataFrame(columns=interests_encoder.classes_)
        interests_df.loc[0] = 0
        for interest in interests:
            if interest in interests_encoder.classes_:
                interests_df.loc[0, interest] = 1
            else:
                print(f"⚠️ Unknown interest: {interest}")  # Handle unknown interests

        print(f"Available education levels: {education_encoder.classes_}")
        education_cleaned = education.strip()
        education_value = education_encoder.transform([education_cleaned])[0]
        education_df = pd.DataFrame([[education_value]], columns=['education_encoded'])

        # Combine features
        X = pd.concat([skills_df.reset_index(drop=True),
                       interests_df.reset_index(drop=True),
                       education_df.reset_index(drop=True)], axis=1)

        # Remove duplicate columns in X
        X = X.loc[:, ~X.columns.duplicated()]

        print("Prediction DataFrame Columns:")
        print(X.columns.tolist())

        print("Feature Names From Training:")
        print(feature_names)
        missing_cols = set(feature_names) - set(X.columns)
        extra_cols = set(X.columns) - set(feature_names)

        if missing_cols:
            print("⚠️ Missing columns in prediction data:", missing_cols)
        if extra_cols:
            print("⚠️ Extra columns in prediction data:", extra_cols)

        X = X.reindex(columns=feature_names, fill_value=0)

        # Final check
        assert list(X.columns) == feature_names, "❌ Feature order mismatch!"


        # Predict
        probabilities = model.predict_proba(X)

        # Top N predictions
        top_indices = np.argsort(probabilities[0])[::-1][:top_n]
        top_probabilities = probabilities[0][top_indices]
        top_careers = target_encoder.inverse_transform(top_indices)

        # Build results
        results = []
        for i, (career, probability) in enumerate(zip(top_careers, top_probabilities)):
            matching_skills = [skill for skill in skills if skill in skills_encoder.classes_]
            matching_interests = [interest for interest in interests if interest in interests_encoder.classes_]

            explanation = {
                "skills": matching_skills[:3],
                "interests": matching_interests[:3],
                "education_match": True  # Simplified
            }

            results.append({
                "rank": i + 1,
                "career": career,
                "confidence": float(probability),
                "explanation": explanation
            })

        return results

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Example usage
    age = 28
    education = "bachelor's"
    skills = ["teamwork", "communication", "public speaking" , "advertising"]
    interests = ["art",  "creativity", "voice acting", "dancing"]

    print(f"Predicting careers for a {age}-year-old with {education}")
    print(f"Skills: {', '.join(skills)}")
    print(f"Interests: {', '.join(interests)}")
    print("\nTop career recommendations:")

    predictions = predict_career(age, education, skills, interests)

    if predictions:
        for pred in predictions:
            print(f"\n{pred['rank']}. {pred['career']} (Confidence: {pred['confidence']:.2f})")
            print(f"   Matching skills: {', '.join(pred['explanation']['skills'])}")
            print(f"   Matching interests: {', '.join(pred['explanation']['interests'])}")
    else:
        print("No predictions could be made. Check the error messages above.")
