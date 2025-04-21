import pandas as pd
import random

# Define some sample skills and descriptions (with renamed skills and additional unique ones)
skills_data = [
    ("Python Programming", "Technical", "The ability to write Python code for various applications."),
    ("Data Analysis", "Analytical", "The ability to analyze and interpret data to make informed decisions."),
    ("Communication", "Soft", "The ability to effectively communicate ideas and information."),
    ("Project Management", "Management", "The ability to plan, execute, and oversee projects from start to finish."),
    ("Creative Thinking", "Creative", "The ability to think outside the box and come up with innovative solutions."),
    ("SQL Databases", "Technical", "The ability to write SQL queries to interact with databases."),
    ("Leadership", "Management", "The ability to lead, motivate, and guide a team towards achieving goals."),
    ("Time Management", "Soft", "The ability to manage and prioritize time effectively."),
    ("Machine Learning", "Technical", "The ability to apply algorithms and statistical models to build predictive models."),
    ("Emotional Intelligence", "Soft", "The ability to recognize, understand, and manage your own emotions, and the emotions of others."),
    ("Business Strategy", "Management", "The ability to create, implement, and analyze business strategies."),
    ("Graphic Design", "Creative", "The ability to create visual content using various design software."),
    ("Problem Solving", "Analytical", "The ability to approach complex problems and find effective solutions."),
    ("Cloud Computing", "Technical", "The ability to manage and utilize cloud infrastructure and services."),
    ("Negotiation", "Soft", "The ability to negotiate with others to reach mutually beneficial agreements."),
    ("Marketing Strategy", "Management", "The ability to plan, execute, and analyze marketing campaigns."),
    ("UI/UX Design", "Creative", "The ability to design user interfaces and optimize user experiences."),
    ("Statistics", "Analytical", "The ability to apply statistical methods and models to data."),
    ("Cybersecurity", "Technical", "The ability to protect networks, devices, and data from cyber threats."),
    ("Presentation Skills", "Soft", "The ability to effectively present information to an audience."),
    ("Agile Methodology", "Management", "The ability to implement and manage Agile frameworks in project management."),
    ("Healthcare Management", "Management", "A professional responsible for managing healthcare systems and teams."),
    ("Healthcare Support", "Technical", "A healthcare professional who provides care to patients and supports doctors."),
    ("Plumbing Systems", "Technical", "A tradesperson responsible for installing and maintaining piping systems."),
    ("Culinary Arts", "Creative", "A culinary professional responsible for preparing food and managing kitchen operations."),
    ("Academic Teaching", "Soft", "An academic professional who teaches courses in a college or university."),
    ("Classroom Management", "Soft", "An education professional who instructs students in a classroom setting."),
    ("Human Resource Management", "Management", "A professional responsible for managing employee relations and organizational development."),
    ("IT Infrastructure", "Technical", "A professional responsible for overseeing and managing IT infrastructure and services."),
    ("Building Maintenance", "Soft", "A person responsible for cleaning and maintaining the cleanliness of buildings and spaces."),
    ("System Maintenance", "Technical", "A professional responsible for maintaining and managing IT systems and servers."),
    ("IT Security and Efficiency", "Analytical", "A professional who evaluates and ensures the security and efficiency of IT systems."),
    ("Music Performance and Production", "Creative", "An artist who creates and performs music, including composing and producing."),
    ("Visual Arts", "Creative", "An artist who creates visual artwork using paint and other materials."),
    ("Construction Management", "Technical", "A professional responsible for constructing buildings and other physical structures."),
    ("Legal Services", "Management", "A legal professional who represents clients in legal matters and court proceedings."),
    ("Performance Arts", "Creative", "An individual who performs in films, television shows, theater, or other media."),
    ("Community Support", "Soft", "A professional who helps individuals and families in need of assistance and support."),
    ("Customer Service", "Soft", "A person who serves food and beverages to customers in a restaurant or other establishment."),
    ("Linguistics and Language Analysis", "Analytical", "A professional who studies language and its structure, development, and usage."),
    ("Geology", "Analytical", "A scientist who studies the Earth's structure, materials, and processes."),
    ("Petroleum Engineering", "Technical", "An engineer responsible for the exploration, extraction, and production of oil and gas."),
    ("Tax Auditing", "Analytical", "A professional who inspects tax records and financial statements for accuracy."),
    ("Accounting", "Analytical", "A financial professional who manages financial records and prepares reports."),
    
    # Additional Unique Skills
    ("Artificial Intelligence", "Technical", "The ability to create systems that mimic human intelligence processes."),
    ("Robotics", "Technical", "The design, construction, and operation of robots."),
    ("Quantum Computing", "Technical", "The ability to work with computers that use quantum-mechanical phenomena."),
    ("Business Analytics", "Analytical", "The ability to use data analysis to improve business decision-making."),
    ("Conflict Resolution", "Soft", "The ability to resolve disputes and conflicts in a constructive manner."),
    ("E-Commerce Strategy", "Management", "The ability to develop strategies for selling products and services online."),
    ("Data Engineering", "Technical", "The ability to design and implement systems for managing large datasets."),
    ("Renewable Energy", "Technical", "The ability to develop and utilize energy from renewable resources."),
    ("Sustainability Practices", "Analytical", "The ability to develop practices that reduce environmental impact."),
    ("Supply Chain Management", "Management", "The management of the flow of goods and services from origin to consumer."),
    ("Digital Marketing", "Management", "The use of digital channels to promote products or services."),
    ("Financial Planning", "Analytical", "The ability to manage and plan finances effectively for long-term success."),
    ("Cyber Forensics", "Technical", "The ability to investigate and analyze digital evidence."),
    ("Blockchain Development", "Technical", "The development of decentralized and distributed ledgers for transactions."),
    ("Data Visualization", "Analytical", "The ability to present data in visual formats like charts and graphs."),
    ("Change Management", "Management", "The ability to manage and facilitate organizational changes effectively."),
    ("Negotiation Skills", "Soft", "The ability to successfully negotiate agreements and resolve conflicts."),
    ("Health Informatics", "Technical", "The use of technology to manage and analyze health data."),
    ("Mobile App Development", "Technical", "The creation of software applications for mobile devices."),
    ("Cloud Security", "Technical", "The ability to protect cloud-based infrastructure and data from security threats."),
    ("Public Speaking", "Soft", "The ability to speak confidently in front of an audience."),
    ("Software Testing", "Technical", "The process of evaluating software to ensure it meets specified requirements."),
    ("Event Planning", "Management", "The ability to plan, organize, and execute events successfully."),
    ("Data Privacy", "Analytical", "The protection of personal information from unauthorized access."),
    ("Customer Relationship Management (CRM)", "Management", "The use of technology to manage and analyze customer interactions."),
    ("Social Media Marketing", "Management", "The use of social media platforms to promote products or services."),
    ("Video Production", "Creative", "The process of creating videos for entertainment, education, or promotion."),
    ("Search Engine Optimization (SEO)", "Analytical", "The process of improving website visibility on search engines."),
    ("Game Development", "Creative", "The process of designing, programming, and testing video games."),
    ("User Research", "Analytical", "The process of studying users to understand their needs and preferences."),
    ("Content Management", "Management", "The process of managing and optimizing digital content."),
    ("Healthcare IT", "Technical", "The use of technology in managing healthcare systems and data."),
    ("Product Development", "Management", "The process of designing, developing, and bringing new products to market."),
    ("Investment Strategy", "Analytical", "The development of plans for managing investments to maximize returns."),
    ("Automation", "Technical", "The use of technology to perform tasks without human intervention."),
    ("Legal Compliance", "Analytical", "The ability to ensure that a business adheres to laws and regulations."),
    ("Brand Management", "Management", "The process of maintaining and improving a company's brand image."),
    ("Customer Experience", "Soft", "The process of ensuring that customers have a positive experience with a company."),
    ("Digital Transformation", "Management", "The integration of digital technology into all areas of business."),
]

# Define the number of records to generate
num_records = 150 # Updated to generate 1500 records

# Randomly select skills and skill types
dataset = []

for _ in range(num_records):
    skill_id = random.randint(1, 150)
    skill_name, skill_type, skill_description = random.choice(skills_data)
    
    dataset.append({
        'skill_id': skill_id,
        'skill_name': skill_name,
        'skill_type': skill_type,
        'skill_description': skill_description
    })

# Create a DataFrame from the dataset
df = pd.DataFrame(dataset)

# Remove duplicates in the dataset to ensure non-redundant entries
df = df.drop_duplicates()

# Save the dataset to a CSV file
df.to_csv('skills_dataset.csv', index=False)

# Show the first few rows of the dataset
print(df.head())
