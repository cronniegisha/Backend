from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Profile, Education, Interest, Job, Skill, ProfileSkill

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

class SkillSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProfileSkill
        fields = [ 'id', 'name', 'level']

class InterestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Interest
        fields = ['id', 'name', 'category']

class EducationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Education
        fields = ['id', 'institution', 'degree', 'field', 'start_year', 'end_year', 'description']

class ProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    skills = SkillSerializer(many=True, read_only=True)
    interests = InterestSerializer(many=True, read_only=True)
    education = EducationSerializer(many=True, read_only=True)
    name = serializers.SerializerMethodField()
    email = serializers.SerializerMethodField()

    class Meta:
        model = Profile
        fields = [
            'id', 'user', 'name', 'email', 'title', 'bio', 'gender', 'age',
            'education_level', 'experience', 'career_preferences',
            'location', 'phone', 'website', 'skills', 'interests', 'education'
        ]
    
    def get_name(self, obj):
        return f"{obj.user.first_name} {obj.user.last_name}".strip() or obj.user.username
    
    def get_email(self, obj):
        return obj.user.email

class ProfileHeaderSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    title = serializers.CharField(max_length=100)
    bio = serializers.CharField(allow_blank=True)
    
    def update(self, instance, validated_data):
        # Update user's name (split into first_name and last_name)
        name_parts = validated_data.pop('name', '').split(' ', 1)
        instance.user.first_name = name_parts[0] if name_parts else ''
        instance.user.last_name = name_parts[1] if len(name_parts) > 1 else ''
        instance.user.save()
        
        # Update profile fields
        instance.title = validated_data.get('title', instance.title)
        instance.bio = validated_data.get('bio', instance.bio)
        instance.save()
        
        return instance

class PersonalInfoSerializer(serializers.Serializer):
    name = serializers.SerializerMethodField()
    email = serializers.SerializerMethodField()  # Change this line too
    title = serializers.CharField(max_length=100)
    bio = serializers.CharField(allow_blank=True)
    gender = serializers.CharField(max_length=20, allow_blank=True)
    age = serializers.IntegerField(allow_null=True, required=False)
    educationLevel = serializers.CharField(source='education_level', max_length=50, allow_blank=True)
    experience = serializers.CharField(max_length=50, allow_blank=True)
    careerPreferences = serializers.CharField(source='career_preferences', max_length=100, allow_blank=True)
    location = serializers.CharField(max_length=100, allow_blank=True)
    phone = serializers.CharField(max_length=20, allow_blank=True)
    website = serializers.URLField(allow_blank=True)
    
    def get_name(self, obj):
        return f"{obj.user.first_name} {obj.user.last_name}".strip() or obj.user.username
    
    def get_email(self, obj):
        return obj.user.email
    
    def update(self, instance, validated_data):
        # The rest of your update method remains unchanged
        name_parts = validated_data.pop('name', '').split(' ', 1)
        instance.user.first_name = name_parts[0] if name_parts else ''
        instance.user.last_name = name_parts[1] if len(name_parts) > 1 else ''
        instance.user.email = validated_data.pop('email', instance.user.email)
        instance.user.save()
        
        # Update profile fields
        instance.title = validated_data.get('title', instance.title)
        instance.bio = validated_data.get('bio', instance.bio)
        instance.gender = validated_data.get('gender', instance.gender)
        instance.age = validated_data.get('age', instance.age)
        instance.education_level = validated_data.get('education_level', instance.education_level)
        instance.experience = validated_data.get('experience', instance.experience)
        instance.career_preferences = validated_data.get('career_preferences', instance.career_preferences)
        instance.location = validated_data.get('location', instance.location)
        instance.phone = validated_data.get('phone', instance.phone)
        instance.website = validated_data.get('website', instance.website)
        instance.save()
        
        return instance

# Fixed serializers for updating collections
class SkillUpdateSerializer(serializers.Serializer):
    skills = serializers.ListField(
        child=serializers.DictField()
    )
    
    def validate_skills(self, value):
        for skill in value:
            if 'name' not in skill or 'level' not in skill:
                raise serializers.ValidationError("Each skill must have 'name' and 'level' fields")
        return value

class InterestUpdateSerializer(serializers.Serializer):
    interests = serializers.ListField(
        child=serializers.DictField()
    )
    
    def validate_interests(self, value):
        for interest in value:
            if 'name' not in interest:
                raise serializers.ValidationError("Each interest must have a 'name' field")
            if 'category' not in interest:
                interest['category'] = 'Personal'  # Default category
        return value

class EducationUpdateSerializer(serializers.Serializer):
    education = serializers.ListField(
        child=serializers.DictField()
    )
    
    def validate_education(self, value):
        for edu in value:
            if 'institution' not in edu or 'degree' not in edu:
                raise serializers.ValidationError("Each education entry must have 'institution' and 'degree' fields")
        return value