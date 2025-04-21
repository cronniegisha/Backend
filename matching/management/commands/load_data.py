# career_match/management/commands/load_data.py

import pandas as pd
from django.core.management.base import BaseCommand
from matching.models import Career  # Import your model

class Command(BaseCommand):
    help = 'Load career data into the database from a CSV file'

    def handle(self, *args, **kwargs):
        # Define the path to your CSV file (adjust this path)
        file_path = "C:/Users/SHIRAH/Desktop/career_matching/career_match/career_data.csv"

        # Read the CSV file
        try:
            career_data = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            self.stderr.write(f"Error reading the CSV file: {str(e)}")
            return
        career_data.columns = career_data.columns.str.strip()
        self.stdout.write(self.style.SUCCESS(f"Columns in CSV: {career_data.columns.tolist()}"))

        # Loop through the data and create Career instances
        for _, row in career_data.iterrows():
            Career.objects.create(
                career_name=row['Career_name'],
                description=row['Description'],
                required_skills=row['Required_skills'],
                qualifications=row['Qualifications'],
                industry_type=row['Industry_type'],
            )

        self.stdout.write(self.style.SUCCESS(f"Successfully loaded career data from {file_path}"))
