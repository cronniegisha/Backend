from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('api/submit-assessment/', views.submit_assessment, name='submit_assessment'),
    path('api/skills/', views.get_skills, name='get_skills'),
    path('recommend-careers/', views.recommend_careers, name='recommend-careers'),
    path('recommend-learning/', views.recommend_learning, name='recommend-learning'),
]
