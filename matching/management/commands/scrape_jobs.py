from django.core.management.base import BaseCommand
from matching.scraper import scrape_brighter_monday
from django_filters.rest_framework import DjangoFilterBackend
from django.core.management.base import BaseCommand
from matching.scraper import scrape_brighter_monday  

class Command(BaseCommand):
    help = "Scrapes BrighterMonday job listings"

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting job scraping...")
        scrape_brighter_monday()
        try:
            jobs_count = scrape_brighter_monday()
            self.stdout.write(self.style.SUCCESS(f'Successfully scraped {jobs_count} new jobs'))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'Error during scraping: {e}'))



