from django.contrib import admin

# Register your models here.


from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Job

@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ('title', 'company', 'location', 'job_type', 'posted_date', 'created_at')
    list_filter = ('job_type', 'location', 'company')
    search_fields = ('title', 'description', 'company')
    date_hierarchy = 'created_at'




