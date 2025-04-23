from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.http import HttpResponse
from .models import UserAssessment
from ai_model.predict import identify_skill_gaps
from ai_model.resources import get_learning_resources
from django.http import JsonResponse
import json
# Add CORS headers to allow requests from your frontend
from django.views.decorators.csrf import csrf_exempt


def home(request):
    """
    A simple home endpoint to welcome users to the system.
    """
    return HttpResponse("Welcome to the E-Career Guidance System!")
@csrf_exempt
def recommend_careers(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_skills = data.get('skills', [])

        # 💡 Your logic to recommend careers based on `user_skills`
        careers = ["Software Developer", "Data Analyst"]  # Replace with actual logic

        return JsonResponse({"careers": careers})
    return JsonResponse({"error": "Invalid request method"}, status=400)

@csrf_exempt
def recommend_learning(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        missing_skills = data.get('missing_skills', [])
        skills_to_improve = data.get('skills_to_improve', [])

        # Define a basic resource map
        resource_map = {
            "Python": "https://learnpython.org/",
            "Data Analysis": "https://www.coursera.org/learn/data-analysis",
            "React": "https://react.dev/learn",
            "Django": "https://docs.djangoproject.com/en/stable/intro/tutorial01/",
            "Problem Solving": "https://www.hackerrank.com/domains/tutorials/10-days-of-javascript"
        }

        # Generate resources separately
        missing_resources = [
            {"skill": skill, "resource": resource_map.get(skill, f"https://www.google.com/search?q=learn+{skill.replace(' ', '+')}")}
            for skill in missing_skills
        ]

        improvement_resources = [
            {"skill": skill, "resource": resource_map.get(skill, f"https://www.google.com/search?q=improve+{skill.replace(' ', '+')}")}
            for skill in skills_to_improve
        ]

        return JsonResponse({
            "missing_resources": missing_resources,
            "improvement_resources": improvement_resources
        })

    return JsonResponse({"error": "Invalid request method"}, status=400)
def process_skills(user_skills):
    """
    Process the user's skills and return analysis results.
    """
    if not user_skills or not isinstance(user_skills, list):
        return {"error": "Invalid or missing 'skills' list."}

    # Step 1: Analyze skills and predict gaps
    skill_analysis = identify_skill_gaps(user_skills)
    
    # Debugging: print the skill analysis
    print("Skill Analysis:", skill_analysis)

    # Step 2: Extract skills by category
    missing_skills = [
        skill.split(": ")[1]
        for skill in skill_analysis if skill.startswith("Missing skill")
    ]
    
    skills_to_improve = [
        skill.split(": ")[1]
        for skill in skill_analysis if skill.startswith("Skill to improve")
    ]
    
    strong_skills = [
        skill.split(": ")[1]
        for skill in skill_analysis if skill.startswith("Strong skill")
    ]

    # Step 3: Fetch learning recommendations
    learning_recommendations = get_learning_resources(missing_skills + skills_to_improve)

    # Step 4: Calculate percentage scores by skill type
    # This is a placeholder - implement your actual percentage calculation logic
    percentage_scores = calculate_percentage_scores(user_skills)

    return {
        "percentage_scores": percentage_scores,
        "strong_skills": strong_skills,
        "skills_to_improve": skills_to_improve,
        "missing_skills": missing_skills,
        "learning_recommendations": learning_recommendations
    }


def calculate_percentage_scores(user_skills):
    """
    Calculate percentage scores by skill type.
    Implement your actual calculation logic from your ML model here.
    """
    # This is a placeholder - replace with your actual calculation
    skill_types = {
        "Technical Skills": [],
        "Soft Skills": [],
        "Management Skills": [],
        "Analytical": [],
        "Creative": []
    }
    
    # Group skills by type
    for skill in user_skills:
        skill_type = skill.get("type", "Technical Skills")
        if skill_type in skill_types:
            skill_types[skill_type].append(skill)
    
    # Calculate average score for each type
    percentage_scores = {}
    for skill_type, skills in skill_types.items():
        if skills:
            avg_score = sum(skill.get("score", 0) for skill in skills) / len(skills)
            # Convert to percentage (assuming score is 1-5)
            percentage = (avg_score / 5) * 100
            percentage_scores[skill_type] = f"{percentage:.1f}%"
        else:
            percentage_scores[skill_type] = "0.0%"
    
    return percentage_scores

@api_view(['POST'])
@permission_classes([AllowAny])  # Allow any client to access this endpoint
@csrf_exempt  # Disable CSRF for this API endpoint
def submit_assessment(request):
    """
    Process skill assessment data and return formatted results for the frontend.
    """
    try:
        # Get skills data from request
        user_skills = request.data.get('skills', [])
        
        # Process the skills
        result = process_skills(user_skills)
        
        if "error" in result:
            return Response({"error": result["error"]}, status=400)
        
        # Format the response for the frontend
        response_data = {
            "percentageScores": result["percentage_scores"],
            "strongSkills": result["strong_skills"],
            "skillsToImprove": [
                {
                    "skill": skill,
                    "course": result["learning_recommendations"].get(skill, {}).get("course", "No course available"),
                    "link": result["learning_recommendations"].get(skill, {}).get("link", "")
                }
                for skill in result["skills_to_improve"]
            ],
            "missingSkills": [
                {
                    "skill": skill,
                    "course": result["learning_recommendations"].get(skill, {}).get("course", "No course available"),
                    "link": result["learning_recommendations"].get(skill, {}).get("link", "")
                }
                for skill in result["missing_skills"]
            ]
        }
        
        # Save assessment to database if needed
        UserAssessment.objects.create(
            user=request.user if request.user.is_authenticated else None,
            skills=user_skills,
            results=response_data
        )
        
        return Response(response_data)

    except Exception as e:
        print(f"Error processing assessment: {str(e)}")
        return Response({"error": str(e)}, status=500)




@api_view(['GET'])
@permission_classes([AllowAny])
def get_skills(request):
    """
    Return a list of predefined skills for the assessment.
    """
    # You can replace this with skills from your database
    skills = [
        {"id": 1, "name": "Python Programming", "type": "Technical Skills"},
        {"id": 2, "name": "Data Analysis", "type": "Technical Skills"},
        {"id": 3, "name": "Communication", "type": "Soft Skills"},
        {"id": 4, "name": "Project Management", "type": "Management Skills"},
        {"id": 5, "name": "Problem Solving", "type": "Analytical"},
        {"id": 6, "name": "Creative Thinking", "type": "Creative"},
        {"id": 7, "name": "JavaScript", "type": "Technical Skills"},
        {"id": 8, "name": "Leadership", "type": "Management Skills"},
        {"id": 9, "name": "Critical Thinking", "type": "Analytical"},
        {"id": 10, "name": "Design Thinking", "type": "Creative"},
    ]
    
    return Response(skills)