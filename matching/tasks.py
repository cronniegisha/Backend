from celery import shared_task
from .scraper import scrape_brighter_monday
import logging

logger = logging.getLogger(__name__)

@shared_task
def scrape_jobs_task():
    """Celery task to scrape jobs from BrighterMonday"""
    logger.info("Starting job scraping task")
    jobs_count = scrape_brighter_monday()
    logger.info(f"Job scraping completed. Added {jobs_count} new jobs.")
    return jobs_count

