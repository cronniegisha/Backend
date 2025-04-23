from django.contrib import admin
from django.urls import path, include
from skill_api.views import home 
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('skill_api.urls')),  # Replace 'your_app' with your actual Django app name
    path('', home),
]
