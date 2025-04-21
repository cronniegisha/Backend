import os
import django
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from .models import Job

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "career_guidance.settings")
django.setup()

def scrape_brighter_monday():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=options)

    scraped_jobs = 0
    base_url = "https://www.brightermonday.co.ug/jobs?page={}"

    for page in range(1, 6):
        print(f"Scraping page {page}...")

        job_url = base_url.format(page)
        driver.get(job_url)
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        job_cards = soup.find_all("div", class_="w-full")

        print(f"Found {len(job_cards)} job cards on page {page}.")

        for card in job_cards:
            try:
                title_tag = card.find("p", class_="text-lg font-medium break-words text-link-500")
                if not title_tag or not title_tag.get_text(strip=True):
                    print("Skipped job with missing title.")
                    continue

                title = title_tag.get_text(strip=True)

                company_tag = card.find("p", class_="text-sm text-link-500 text-loading-animate inline-block")
                company = company_tag.get_text(strip=True) if company_tag else "No Company"

                location_tag = card.find("span", class_="mb-3 px-3 py-1 rounded bg-brand-secondary-100 mr-2 text-loading-hide")
                location = location_tag.get_text(strip=True) if location_tag else "No Location"

                job_url_tag = title_tag.find_parent("a")
                job_url = (
                    job_url_tag['href']
                    if job_url_tag and job_url_tag.has_attr('href') else None
                )

                if not job_url:
                    print(f"Skipped job '{title}' due to missing job URL.")
                    continue

                job_type_tag = card.find("a", class_="text-xs bg-neutral-100")
                job_type = job_type_tag.get_text(strip=True) if job_type_tag else "Not Specified"

                posted_date_tag = card.find("p", class_="text-xs text-neutral-500")
                posted_date = posted_date_tag.get_text(strip=True) if posted_date_tag else "Unknown"

                # Skip duplicates
                if Job.objects.filter(title=title, company=company, location=location, job_url=job_url).exists():
                   
                    continue

                Job.objects.create(
                    title=title,
                    company=company,
                    location=location,
                    job_url=job_url,
                    job_type=job_type,
                    posted_date=posted_date,
                )
                scraped_jobs += 1
                print(f"Saved: {title}")

            except Exception as e:
                print("Error parsing job card:", e)

    driver.quit()
    print(f"Successfully scraped {scraped_jobs} job(s)")
