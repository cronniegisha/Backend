# 🧠 E-Career Guidance System (ECGS)

**E-Career Guidance System** is a web-based platform built with Django REST Framework and React to help users in Uganda explore career paths based on personal information, and view active job opportunities scraped from BrighterMonday.

---

## 🚀 Features

- 🔍 **Automated Job Scraping** from [BrighterMonday Uganda](https://www.brightermonday.co.ug/jobs)
- 📊 AI-informed **career guidance**
- 🕒 Scheduled scraping using **Celery + Beat**

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/Kyavawa5Shidah/Backend.git
```

### 2. set up virtual environment
```bash
python -m venv .venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Backend set-up
# Configure database in settings.py
# Run initial setup
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```






