from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Profile, Education, Interest, Job, Skill, ProfileSkill, CareerPrediction, CareerResult

User = get_user_model()


class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = '__all__'


class SkillSerializer(serializers.ModelSerializer):
    class Meta:
        model = Skill
        fields = "__all__"


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def create(self, validated_data):
        print("Creating user...")
        password = validated_data.pop('password')  # Take out password
        user = User(**validated_data)
        user.set_password(password)  # 🔐 Hash the password
        user.save()
        return user

# Update the SkillSerializer to match the ProfileSkill model
class SkillSerializer(serializers.ModelSerializer):
    id = serializers.CharField(required=False)  # Allow client-side IDs
    
    class Meta:
        model = ProfileSkill
        fields = ['id', 'name', 'level']
        
    def to_internal_value(self, data):
        # Handle client-side IDs by removing them before validation
        if isinstance(data, dict) and 'id' in data and not data['id'].isdigit():
            data = data.copy()
            data.pop('id', None)
        return super().to_internal_value(data)

class InterestSerializer(serializers.ModelSerializer):
    id = serializers.CharField(required=False)  # Allow client-side IDs
    
    class Meta:
        model = Interest
        fields = ['id', 'name', 'category']
        
    def to_internal_value(self, data):
        # Handle client-side IDs by removing them before validation
        if isinstance(data, dict) and 'id' in data and not data['id'].isdigit():
            data = data.copy()
            data.pop('id', None)
        return super().to_internal_value(data)

class EducationSerializer(serializers.ModelSerializer):
    id = serializers.CharField(required=False)  # Allow client-side IDs
    
    class Meta:
        model = Education
        fields = ['id', 'institution', 'degree', 'field', 'year', 'description']
        
    def to_internal_value(self, data):
        # Handle client-side IDs by removing them before validation
        if isinstance(data, dict) and 'id' in data and not data['id'].isdigit():
            data = data.copy()
            data.pop('id', None)
        return super().to_internal_value(data)

class ProfileSerializer(serializers.ModelSerializer):
    skills = SkillSerializer(many=True, required=False)
    interests = InterestSerializer(many=True, required=False)
    education = EducationSerializer(many=True, required=False)
    username = serializers.CharField(source='user.username', read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)
    
    class Meta:
        model = Profile
        fields = [
            'id', 'username', 'email', 'name', 'title', 'bio', 'gender', 
            'age', 'education_level', 'experience', 'career_preferences', 
            'location', 'phone', 'website', 'image', 'skills', 'interests', 
            'education', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def create(self, validated_data):
        skills_data = validated_data.pop('skills', [])
        interests_data = validated_data.pop('interests', [])
        education_data = validated_data.pop('education', [])
        
        profile = Profile.objects.create(**validated_data)
        
        for skill_data in skills_data:
            ProfileSkill.objects.create(profile=profile, **skill_data)
        
        for interest_data in interests_data:
            Interest.objects.create(profile=profile, **interest_data)
        
        for edu_data in education_data:
            Education.objects.create(profile=profile, **edu_data)
        
        return profile
    
    def update(self, instance, validated_data):
        # Update profile fields
        for attr, value in validated_data.items():
            if attr not in ['skills', 'interests', 'education']:
                setattr(instance, attr, value)
        instance.save()
        
        return instance

class PersonalInfoSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(source='user.email', read_only=True)
    
    class Meta:
        model = Profile
        fields = [
            'name', 'email', 'title', 'bio', 'gender', 'age', 
            'education_level', 'experience', 'career_preferences', 
            'location', 'phone', 'website'
        ]



class CareerResultSerializer(serializers.ModelSerializer):
    """
    Serializer for individual career results.
    """
    requiredSkills = serializers.JSONField(source='required_skills')
    matchScore = serializers.IntegerField(source='match_score')
    industryType = serializers.CharField(source='industry_type')
    
    class Meta:
        model = CareerResult
        fields = ['id', 'title', 'matchScore', 'description', 'industryType', 'requiredSkills']


class CareerPredictionSerializer(serializers.ModelSerializer):
    """
    Serializer for career predictions, including nested results.
    """
    results = CareerResultSerializer(many=True, read_only=True)
    explanation = serializers.SerializerMethodField()
    
    class Meta:
        model = CareerPrediction
        fields = ['id', 'created_at', 'explanation', 'results']
    
    def get_explanation(self, obj):
        """
        Format the explanation data to match the frontend expectations.
        """
        return {
            'skills': obj.skills,
            'interests': obj.interests,
            'education_match': obj.education_match
        }


class SaveCareerPredictionSerializer(serializers.Serializer):
    """
    Serializer for saving career predictions from the frontend.
    """
    educationLevel = serializers.CharField(required=True)
    explanation = serializers.DictField(required=True)
    results = serializers.ListField(required=True)
    
    def create(self, validated_data):
        user = self.context['request'].user
        explanation = validated_data.get('explanation', {})
        results_data = validated_data.get('results', [])
        
        # Create the prediction
        prediction = CareerPrediction.objects.create(
            user=user,
            skills=explanation.get('skills', []),
            interests=explanation.get('interests', []),
            education_match=explanation.get('education_match', False),
            education_level=validated_data.get('educationLevel', '')
        )
        
        # Create the results
        for result_data in results_data:
            CareerResult.objects.create(
                prediction=prediction,
                title=result_data.get('title', ''),
                match_score=result_data.get('matchScore', 0),
                description=result_data.get('description', ''),
                industry_type=result_data.get('industryType', ''),
                required_skills=result_data.get('requiredSkills', [])
            )
        
        return prediction
