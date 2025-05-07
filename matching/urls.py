from .views import register_user, get_brighter_monday_jobs
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .models import *
from .serializers import *
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import SignUpView, verify_token, SignInView, get_csrf_token, logout_view, ensure_user_profile, UserProfileView, current_user, UserSkillsView, education, UserInterestsView, TrackUserActivity, dashboard_summary, user_engagement_summary, export_prediction_report
from .views import (
    EnsureProfileView, ProfileView, ProfileHeaderView, PersonalInfoView,
    SkillsView, InterestsView, EducationView)

urlpatterns = [
    path('trigger-scraping/', views.trigger_scraping, name='trigger-scraping'),
    path('jobs/', get_brighter_monday_jobs, name='brightermonday-jobs'),
    path('register/', register_user, name='register'),
    path('login/', TokenObtainPairView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/submit-assessment/', views.submit_assessment, name='submit_assessment'),
    path('api/skills/', views.get_skills, name='get_skills'),
    path('recommend-careers/', views.recommend_careers, name='recommend-careers'),
    path('recommend-learning/', views.recommend_learning, name='recommend-learning'),
    path('', views.home, name='home'),
    path('predict/', views.predict_career, name='predict_career'),
    path('signup/', SignUpView.as_view(), name='signup'),
    path('signin/', SignInView.as_view(), name='signin'),
    path('track_activity/', TrackUserActivity.as_view(), name='track_activity'),
    path('dashboard_summary/', dashboard_summary, name='dashboard_summary'),
    path('export_prediction_report/', export_prediction_report, name='export_prediction_report'),
    path('user_engagement_summary/', user_engagement_summary, name='user_engagement_summary'),
    path('me/', current_user, name='current_user'),
    path("logout/", logout_view, name="logout"),
    path('verify-token', views.verify_token, name='verify-token'),
    path("csrf/", get_csrf_token),
    path('ensure-profile/', EnsureProfileView.as_view(), name='ensure-profile'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('header/', ProfileHeaderView.as_view(), name='profile-header'),
    path('personal-info/', PersonalInfoView.as_view(), name='personal-info'),
    path('skills/', SkillsView.as_view(), name='skills'),
    path('interests/', InterestsView.as_view(), name='interests'),
    path('education/', EducationView.as_view(), name='education'),
    
   
]
