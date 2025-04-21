# ai_model/model.py

# ai_model/model.py

from ai_model.resources import get_learning_resources

def assess_skills(user_skills):
    """
    Assess the user's skills and provide a score, missing skills, strong skills,
    skills to improve, and learning recommendations.
    """
    if not user_skills or not isinstance(user_skills, list):
        return {
            "error": "Invalid or missing 'skills' list."
        }

    # Predefined skill score mapping (example logic)
    skill_scores = {
        "Python Programming": 90,
        "Git Version Control": 85,
        "Data Analysis": 55,
        "SQL Databases": 50,
        "Marine Engineering": 30,
        "Aerospace Engineering": 40,
        "Communication": 65,
        "Public Speaking": 45,
        "Machine Learning": 80,
        "Software Testing": 78,
    }

    strong_skills = []
    skills_to_improve = []
    missing_skills = []

    for skill in user_skills:
        score = skill_scores.get(skill, 0)

        if score >= 80:
            strong_skills.append(skill)
        elif 60 <= score < 80:
            skills_to_improve.append(skill)
        else:
            missing_skills.append(skill)

    total_skills = len(user_skills)
    total_score = sum(skill_scores.get(skill, 0) for skill in user_skills)
    percentage_score = (total_score / (total_skills * 100)) * 100 if total_skills else 0

    # Fetch resources
    learning_recommendations = get_learning_resources(missing_skills + skills_to_improve)

    return {
        "percentage_score": percentage_score,
        "missing_skills": missing_skills,
        "strong_skills": strong_skills,
        "skills_to_improve": skills_to_improve,
        "learning_recommendations": learning_recommendations
    }




