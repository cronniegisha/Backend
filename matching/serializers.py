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


class InterestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Interest
        fields = ['id', 'name', 'category']

class EducationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Education
        fields = '__all__'
        read_only_fields = ['user']

class SkillSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProfileSkill
        fields = ['id', 'name', 'level', 'description']

class PersonalSerializer(serializers.ModelSerializer):
    name = serializers.CharField(source="user.username", read_only=True)
    email = serializers.EmailField(source="user.email", read_only=True)

    class Meta:
        model = Profile
        fields = [
            'name', 'email', 'title', 'bio', 'gender', 'age',
            'education_level', 'experience', 'career_preferences',
            'location', 'phone', 'website'
        ]

class ProfileSerializer(serializers.ModelSerializer):
    name = serializers.CharField(source='user.username')
    email = serializers.EmailField(source='user.email', read_only=True)

    class Meta:
        model = Profile
        fields = ['name', 'email', 'title', 'bio']

    def update(self, instance, validated_data):
        user_data = validated_data.pop('user', {})
        if 'username' in user_data:
            instance.user.username = user_data['username']
            instance.user.save()
        return super().update(instance, validated_data)

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email')