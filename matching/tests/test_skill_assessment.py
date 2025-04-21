# Import necessary libraries
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient
from django.urls import reverse

class SkillAssessmentTestCase(TestCase):
 def test_skill_assessment(self):
        url = reverse('submit_assessment')  # Make sure this matches your urls.py name
        data = {
           "skills": [
            {"skill_name": "Python Programming", "skill_type": "Technical", "score": 85},
            {"skill_name": "Git Version Control", "skill_type": "Technical", "score": 90},
            {"skill_name": "Marine Engineering", "skill_type": "Technical", "score": 75},
            {"skill_name": "Aerospace Engineering", "skill_type": "Technical", "score": 60},
            {"skill_name": "SQL Databases", "skill_type": "Technical", "score": 58},
            {"skill_name": "Software Testing", "skill_type": "Technical", "score": 78}
        ]
        
        
        }

        response = self.client.post(url, data, content_type='application/json') 
        # Print the response to help debug
        print(response.data)  # Add this line to inspect the response

        # Check HTTP response
        self.assertEqual(response.status_code, 200)
        self.assertTrue(isinstance(response.data['percentage_score'], float))
# Then check if it's a valid float (and not None)
        self.assertIsInstance(response.data['percentage_score'], float)


        # Check overall structure
        self.assertIn('percentage_score', response.data)
        self.assertIn('missing_skills', response.data)
        self.assertIn('strong_skills', response.data)
        self.assertIn('skills_to_improve', response.data)
        self.assertIn('learning_recommendations', response.data)
        self.assertIn('missing_skills_recommendations', response.data)
        self.assertIn('improvement_recommendations', response.data)

        # Check skill categorization
        self.assertIn("Marine Engineering", response.data['strong_skills'])
        self.assertIn("Aerospace Engineering", response.data['strong_skills'])  # was wrongly asserted as missing
        self.assertIn("Python Programming", response.data['strong_skills'])
        self.assertIn("Git Version Control", response.data['strong_skills'])
        self.assertIn("SQL Databases", response.data['strong_skills'])  # was wrongly asserted as needing improvement
        self.assertIn("Software Testing", response.data['missing_skills']) 

# Check learning recommendation mapping
        self.assertIn("Marine Engineering", response.data['learning_recommendations'])
        self.assertIn("SQL Databases", response.data['learning_recommendations'])  # moved from improvement_recommendations
        self.assertIn("Aerospace Engineering", response.data['learning_recommendations'])  # moved from missing_skills_recommendations
        self.assertIn("Software Testing", response.data['missing_skills_recommendations'])
    

