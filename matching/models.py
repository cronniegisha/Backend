
from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver




class Job(models.Model):
    title = models.CharField(max_length=255)
    company = models.CharField(max_length=255, null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    posted_date = models.CharField(max_length=100, null=True, blank=True)
    job_url = models.URLField(unique=True, null=True, blank=True) 
    deadline = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    job_type =  models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return f"{self.title} at {self.company}"

    class Meta:
        ordering = ['-created_at']
        db_table = 'joblistings'  



class Skill(models.Model):
    skill_id = models.AutoField(primary_key=True)
    skill_name = models.CharField(max_length=255)
    skill_type = models.CharField(max_length=100)
    skill_description = models.TextField()

    def __str__(self):
        return self.skill_name

class UserAssessment(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE)
    score = models.FloatField()

    def __str__(self):
        return f"User {self.user_id} - {self.skill.skill_name}"


class Career(models.Model):
    career_name = models.CharField(max_length=255)
    description = models.TextField()
    required_skills = models.TextField()
    qualifications = models.TextField()
    industry_type = models.CharField(max_length=100)

    def __str__(self):
        return self.career_name

class CareerMatch(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    career_name = models.CharField(max_length=255)
    description = models.TextField()
    required_skills = models.TextField()
    industry_type = models.CharField(max_length=100)
    match_score = models.FloatField()

    def __str__(self):
        return f"{self.user.username} - {self.career_name}"


class UserManager(BaseUserManager):
    def create_user(self, username, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(username=username, email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(username, email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']


    def __str__(self):
        return self.username


class UserActivity(models.Model):
    session_id = models.CharField(max_length=255)
    event_type = models.CharField(max_length=255)
    event_data = models.JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)

class PredictionHistory(models.Model):
    session_id = models.CharField(max_length=255)
    user_input = models.JSONField()
    predicted_careers = models.JSONField()
    confidence_scores = models.JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)

class AIModelPerformance(models.Model):
    prediction_success = models.BooleanField()
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class Interest(models.Model):
    CATEGORY_CHOICES = [
        ("Personal", "Personal"),
        ("Professional", "Professional"),
        ("Hobby", "Hobby"),
        ("Academic", "Academic"),
        ("Other", "Other"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="interests", null=True, blank=True)
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.category})"


class Education(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='educations', null=True)
    institution = models.CharField(max_length=255)
    degree = models.CharField(max_length=255)
    field = models.CharField(max_length=255, blank=True, null=True)
    start_year = models.CharField(max_length=4, blank=True, null=True)
    end_year = models.CharField(max_length=4, blank=True, null=True)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.degree} at {self.institution}"

class ProfileSkill(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='skills')
    name = models.CharField(max_length=100)
    level = models.CharField(
        max_length=50,
        choices=[
            ('Beginner', 'Beginner'),
            ('Intermediate', 'Intermediate'),
            ('Advanced', 'Advanced'),
            ('Expert', 'Expert')
        ],
        default='Beginner'
    )
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.name} ({self.level})"

class Personal(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='personal_profile')
    title = models.CharField(max_length=100, blank=True, null=True)
    bio = models.TextField(max_length=255, blank=True, null=True)
    gender = models.CharField(max_length=20, blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    education_level = models.CharField(max_length=100, blank=True, null=True)
    experience = models.CharField(max_length=255, blank=True, null=True)
    career_preferences = models.CharField(max_length=255, blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    website = models.URLField(blank=True, null=True)

    def __str__(self):
        return f"Profile of {self.user.username}"

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="main_profile")
    title = models.CharField(max_length=255, blank=True)
    bio = models.TextField(max_length=255, blank=True)

    def __str__(self):
        return f"{self.user.username}'s profile"

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    else:
        instance.profile.save()