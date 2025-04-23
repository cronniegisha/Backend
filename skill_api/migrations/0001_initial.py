from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Skill',
            fields=[
                ('skill_id', models.AutoField(primary_key=True, serialize=False)),
                ('skill_name', models.CharField(max_length=255)),
                ('skill_type', models.CharField(max_length=100)),
                ('skill_description', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='UserAssessment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.IntegerField()),
                ('score', models.FloatField()),
                ('skill', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='skill_api.skill')),  # Replace with correct app label
            ],
        ),
    ]
