from django.db import models

class Skill(models.Model):
    skill_id = models.AutoField(primary_key=True)
    skill_name = models.CharField(max_length=255)
    skill_type = models.CharField(max_length=100)
    skill_description = models.TextField()

    def __str__(self):
        return self.skill_name

class UserAssessment(models.Model):
    user_id = models.IntegerField()
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE)
    score = models.FloatField()

    def __str__(self):
        return f"User {self.user_id} - {self.skill.skill_name}"
