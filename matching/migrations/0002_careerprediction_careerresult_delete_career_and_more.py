# Generated by Django 5.1.7 on 2025-05-10 04:29

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('matching', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CareerPrediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('skills', models.JSONField(default=list)),
                ('interests', models.JSONField(default=list)),
                ('education_match', models.BooleanField(default=False)),
                ('education_level', models.CharField(blank=True, max_length=100)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='career_predictions', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='CareerResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('match_score', models.IntegerField()),
                ('description', models.TextField(blank=True)),
                ('industry_type', models.CharField(blank=True, max_length=255)),
                ('required_skills', models.JSONField(default=list)),
                ('prediction', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='results', to='matching.careerprediction')),
            ],
            options={
                'ordering': ['-match_score'],
            },
        ),
        migrations.DeleteModel(
            name='Career',
        ),
        migrations.DeleteModel(
            name='CareerMatch',
        ),
    ]
