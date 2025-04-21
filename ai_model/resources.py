import os
import joblib



# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODER_DIR = os.path.join(BASE_DIR, 'skill_assessment', 'encoders')

skill_name_encoder_path = os.path.join(os.path.dirname(__file__), '..', 'skill_assessment', 'encoders', 'label_encoder_skill_name.pkl')
skill_type_encoder_path = os.path.join(os.path.dirname(__file__), '..', 'skill_assessment', 'encoders', 'label_encoder_skill_type.pkl')

# Error handling for missing encoder files
if not os.path.exists(skill_name_encoder_path):
    raise FileNotFoundError(f"Skill name encoder file not found at {skill_name_encoder_path}")
if not os.path.exists(skill_type_encoder_path):
    raise FileNotFoundError(f"Skill type encoder file not found at {skill_type_encoder_path}")


# Learning resources dictionary with clickable Markdown links
LEARNING_RESOURCES = {
    "Game Development": {
        "course": "Introduction to Game Development",
        "link": "https://www.coursera.org/learn/game-development"
    },
    "Negotiation": {
        "course": "Successful Negotiation: Essential Strategies and Skills",
        "link": "https://www.coursera.org/learn/negotiation"
    },
    "Linguistics and Language Analysis": {
        "course": "Miracles of Human Language: An Introduction to Linguistics",
        "link": "https://www.coursera.org/learn/human-language"
    },
    "Change Management": {
        "course": "Organizational Change and Culture for Managers",
        "link": "https://www.edx.org/course/organizational-change-and-culture-for-managers"
    },
    "Software Testing": {
        "course": "Software Testing and Automation",
        "link": "https://www.coursera.org/specializations/software-testing-automation"
    },
    "Presentation Skills": {
        "course": "Presentation Skills: Speechwriting, Slides and Delivery",
        "link": "https://www.coursera.org/learn/presentation-skills"
    },
    "Robotics": {
        "course": "Modern Robotics: Mechanics, Planning, and Control",
        "link": "https://www.coursera.org/specializations/modernrobotics"
    },
    "SQL Databases": {
        "course": "Databases and SQL for Data Science",
        "link": "https://www.coursera.org/learn/sql-data-science"
    },
    "Customer Experience": {
        "course": "Customer Experience Management",
        "link": "https://www.coursera.org/learn/customer-experience-management"
    },
    "Financial Planning": {
        "course": "Financial Planning for Young Adults",
        "link": "https://www.coursera.org/learn/financial-planning"
    },
    "Project Management": {
        "course": "Introduction to Project Management",
        "link": "https://www.coursera.org/learn/project-management-principles"
    },
    "Graphic Design": {
        "course": "Graphic Design Specialization",
        "link": "https://www.coursera.org/specializations/graphic-design"
    },
    "Content Writing": {
        "course": "Good with Words: Writing and Editing",
        "link": "https://www.coursera.org/learn/writing-editing"
    },
    "Social Media Marketing": {
        "course": "Social Media Marketing Specialization",
        "link": "https://www.coursera.org/specializations/social-media-marketing"
    },
    "Data Analysis": {
        "course": "Data Analysis with Python",
        "link": "https://www.coursera.org/learn/data-analysis-with-python"
    },
    "Sales Strategy": {
        "course": "Sales Training: Techniques for a Human-Centric Sales Process",
        "link": "https://www.coursera.org/learn/human-centric-sales"
    },
    "IT Support": {
        "course": "Google IT Support Professional Certificate",
        "link": "https://www.coursera.org/professional-certificates/google-it-support"
    },
    "Cybersecurity": {
        "course": "Introduction to Cyber Security",
        "link": "https://www.coursera.org/learn/intro-cyber-security"
    },
    "Cloud Computing": {
        "course": "Cloud Computing Basics (Cloud 101)",
        "link": "https://www.coursera.org/learn/cloud-computing-basics"
    },
    "UI/UX Design": {
        "course": "Google UX Design Professional Certificate",
        "link": "https://www.coursera.org/professional-certificates/google-ux-design"
    },
    "Blockchain": {
        "course": "Blockchain Basics",
        "link": "https://www.coursera.org/learn/blockchain-basics"
    },
    "Video Editing": {
        "course": "Video Editing with Adobe Premiere Pro",
        "link": "https://www.coursera.org/projects/video-editing-premiere-pro"
    },
    "Network Security": {
        "course": "Introduction to Network Security",
        "link": "https://www.coursera.org/learn/network-security"
    },
    "Digital Marketing": {
        "course": "Digital Marketing Specialization",
        "link": "https://www.coursera.org/specializations/digital-marketing"
    },
    "Data Visualization": {
        "course": "Data Visualization with Tableau",
        "link": "https://www.coursera.org/learn/data-visualization-tableau"
    },
    "Financial Analysis": {
        "course": "Introduction to Financial Accounting",
        "link": "https://www.coursera.org/learn/wharton-accounting"
    },
    "Agile Methodology": {
        "course": "Agile Development Specialization",
        "link": "https://www.coursera.org/specializations/agile-development"
    },
    "Mobile App Development": {
        "course": "Build Your First Android App (Project-Centered Course)",
        "link": "https://www.coursera.org/learn/android-programming"
    },
     "Time Management": {
        "course": "Work Smarter, Not Harder: Time Management for Personal & Professional Productivity",
        "link": "https://www.coursera.org/learn/work-smarter-not-harder"
    },
    "Customer Service": {
        "course": "Customer Service Fundamentals",
        "link": "https://www.coursera.org/learn/customer-service-fundamentals"
    },
    "Critical Thinking": {
        "course": "Mindware: Critical Thinking for the Information Age",
        "link": "https://www.coursera.org/learn/mindware"
    },
    "Business Analysis": {
        "course": "Business Analysis & Process Management",
        "link": "https://www.coursera.org/learn/business-analysis-process-management"
    },
    "Entrepreneurship": {
        "course": "Entrepreneurship: Launching an Innovative Business",
        "link": "https://www.coursera.org/specializations/wharton-entrepreneurship"
    },
    "Big Data": {
        "course": "Big Data Specialization",
        "link": "https://www.coursera.org/specializations/big-data"
    },
    "Virtual Assistance": {
        "course": "Become a Virtual Assistant",
        "link": "https://www.udemy.com/course/virtual-assistant-training/"
    },
    "Business Communication": {
        "course": "Business Communication",
        "link": "https://www.coursera.org/learn/business-communication"
    },
     "Public Speaking": {
        "course": "Dynamic Public Speaking",
        "link": "https://www.coursera.org/specializations/public-speaking"
    },
    "Data Engineering": {
        "course": "Data Engineering, Big Data, and Machine Learning on GCP",
        "link": "https://www.coursera.org/learn/gcp-data-machine-learning"
    },
    "Foreign Language - French": {
        "course": "Learn French for Global Communication",
        "link": "https://www.duolingo.com/course/fr/en/Learn-French"
    },
    "Digital Illustration": {
        "course": "Digital Illustration for Beginners in Procreate",
        "link": "https://www.skillshare.com/en/classes/Digital-Illustration-for-Beginners-in-Procreate/1103578521"
    },
    "HR Management": {
        "course": "Human Resource Management: HR for People Managers",
        "link": "https://www.coursera.org/specializations/human-resource-management"
    },
    "Robotic Process Automation (RPA)": {
        "course": "Introduction to Robotic Process Automation (RPA)",
        "link": "https://www.coursera.org/learn/introduction-robotic-process-automation-rpa"
    },
"Python Programming": {
        "course": "Python for Everybody",
        "link": "https://www.coursera.org/specializations/python"
    },
    "Machine Learning": {
        "course": "Machine Learning by Stanford",
        "link": "https://www.coursera.org/learn/machine-learning"
    },
    "Communication Skills": {
        "course": "Improving Communication Skills",
        "link": "https://www.coursera.org/learn/wharton-communication-skills"
    },
    "Project Management": {
        "course": "Google Project Management",
        "link": "https://www.coursera.org/professional-certificates/google-project-management"
    },
    "JavaScript Programming": {
        "course": "JavaScript, jQuery, and JSON",
        "link": "https://www.coursera.org/learn/javascript-jquery-json"
    },
    "SQL Databases": {
        "course": "Databases and SQL for Data Science",
        "link": "https://www.coursera.org/learn/sql-for-data-science"
    },
    "Social Work": {
        "course": "Social Work Practice: Introduction to Social Work",
        "link": "https://www.edx.org/course/social-work-practice-introduction-to-social-work"
    },
    "Event Planning": {
        "course": "Event Planning and Management",
        "link": "https://www.udemy.com/course/event-planning-and-management/"
    },
    "AI Ethics": {
        "course": "Ethics of AI and Big Data",
        "link": "https://www.edx.org/course/ethics-of-ai-and-big-data"
    },
    "Material Science": {
        "course": "Materials Science: 10 Things Every Engineer Should Know",
        "link": "https://www.edx.org/course/materials-science-10-things-every-engineer-should-know"
    },
    "Actuarial Science": {
        "course": "Actuarial Science: An Introduction",
        "link": "https://www.coursera.org/learn/actuarial-science-introduction"
    },
    "Emotional Intelligence": {
        "course": "Emotional Intelligence at Work",
        "link": "https://www.udemy.com/course/emotional-intelligence-at-work/"
    },
    "Building Maintenance": {
        "course": "Building Maintenance Fundamentals",
        "link": "https://www.coursera.org/learn/building-maintenance-fundamentals"
    },
    "Healthcare Management": {
        "course": "Healthcare Management Specialization",
        "link": "https://www.coursera.org/specializations/healthcare-management"
    },
    "Customer Relationship Management (CRM)": {
        "course": "Customer Relationship Management (CRM) Fundamentals",
        "link": "https://www.udemy.com/course/customer-relationship-management-crm-fundamentals/"
    },
    "Creative Thinking": {
        "course": "Creative Thinking: Techniques and Tools for Success",
        "link": "https://www.udemy.com/course/creative-thinking-techniques-and-tools-for-success/"
    },
    "Search Engine Optimization (SEO)": {
        "course": "SEO Training Course by Moz",
        "link": "https://moz.com/learn/seo"
    },
    "Conflict Resolution": {
        "course": "Conflict Resolution Skills",
        "link": "https://www.coursera.org/learn/conflict-resolution"
    },
    "Academic Teaching": {
        "course": "Introduction to Academic Teaching",
        "link": "https://www.edx.org/course/introduction-to-academic-teaching"
    },
    "Visual Arts": {
        "course": "Introduction to Visual Arts",
        "link": "https://www.coursera.org/learn/intro-to-visual-arts"
    },
    "Brand Management": {
        "course": "Brand Management: Aligning Business, Brand and Behaviour",
        "link": "https://www.coursera.org/learn/brand-management"
    },
    "Legal Services": {
        "course": "Legal Services and Client Management",
        "link": "https://www.udemy.com/course/legal-services-and-client-management/"
    },
    "Digital Transformation": {
        "course": "Digital Transformation and the IT Team",
        "link": "https://www.coursera.org/learn/digital-transformation-it-team"
    },
    "Conflict Mediation": {
        "course": "Mediation and Conflict Resolution",
        "link": "https://www.udemy.com/course/mediation-and-conflict-resolution/"
    },
    "Lean Management": {
        "course": "Lean Management: The Complete Course",
        "link": "https://www.udemy.com/course/lean-management-complete-course/"
    },
    "Penetration Testing": {
        "course": "Penetration Testing and Ethical Hacking",
        "link": "https://www.udemy.com/course/penetration-testing-and-ethical-hacking/"
    },
    "Customer Journey Mapping": {
        "course": "Customer Journey Mapping for Beginners",
        "link": "https://www.udemy.com/course/customer-journey-mapping-for-beginners/"
    },
    "Organizational Development": {
        "course": "Organizational Development Foundations",
        "link": "https://www.coursera.org/learn/organizational-development-foundations"
    },
    "Conflict De-escalation": {
        "course": "Conflict De-escalation Techniques",
        "link": "https://www.udemy.com/course/conflict-de-escalation-techniques/"
    },
    "Behavioral Analysis": {
        "course": "Behavioral Psychology and Analysis",
        "link": "https://www.udemy.com/course/behavioral-psychology-and-analysis/"
    },
    "Presentation Design": {
        "course": "Presentation Design Masterclass",
        "link": "https://www.udemy.com/course/presentation-design-masterclass/"
    },
    "UI Prototyping": {
        "course": "UI/UX Design: Prototyping with Figma",
        "link": "https://www.udemy.com/course/uiux-design-prototyping-with-figma/"
    },
    "Team Leadership": {
        "course": "Leadership and Team Management",
        "link": "https://www.coursera.org/learn/team-leadership"
    },
    "CI/CD Implementation": {
        "course": "CI/CD Pipeline Fundamentals",
        "link": "https://www.coursera.org/learn/cicd-pipeline"
    },
    "Storyboarding": {
        "course": "Storyboarding for Animation and Film",
        "link": "https://www.udemy.com/course/storyboarding-for-animation-and-film/"
    },
    "Cross-functional Team Management": {
        "course": "Managing Cross-Functional Teams",
        "link": "https://www.udemy.com/course/managing-cross-functional-teams/"
    },
    "Data Modeling": {
        "course": "Data Modeling and Database Design",
        "link": "https://www.udemy.com/course/data-modeling-and-database-design/"
    },
    "Workforce Planning": {
        "course": "Workforce Planning and Optimization",
        "link": "https://www.coursera.org/learn/workforce-planning"
    },
    "Brainstorming Facilitation": {
        "course": "Facilitating Effective Brainstorming Sessions",
        "link": "https://www.udemy.com/course/facilitating-effective-brainstorming-sessions/"
    },
    "Revenue Forecasting": {
        "course": "Revenue Forecasting for Managers",
        "link": "https://www.coursera.org/learn/revenue-forecasting"
    },
    "Customer Empathy": {
        "course": "Customer Empathy and Experience Design",
        "link": "https://www.udemy.com/course/customer-empathy-and-experience-design/"
    },
    "Security Auditing": {
        "course": "Introduction to Security Auditing",
        "link": "https://www.coursera.org/learn/security-auditing"
    },
    "Persona Development": {
        "course": "Persona Development for UX Design",
        "link": "https://www.udemy.com/course/persona-development-for-ux-design/"
    },
    "Brand Identity Design": {
        "course": "Creating a Strong Brand Identity",
        "link": "https://www.udemy.com/course/creating-a-strong-brand-identity/"
    },
    "API Integration": {
        "course": "API Integration and Development",
        "link": "https://www.udemy.com/course/api-integration-and-development/"
    },
    "Data Storytelling": {
        "course": "Data Storytelling for Business",
        "link": "https://www.udemy.com/course/data-storytelling-for-business/"
    },
    "Performance Evaluation": {
        "course": "Employee Performance Evaluation",
        "link": "https://www.udemy.com/course/employee-performance-evaluation/"
    },
    "Containerization": {
        "course": "Docker and Kubernetes for Beginners",
        "link": "https://www.udemy.com/course/docker-and-kubernetes-for-beginners/"
    },
    "Cohort Analysis": {
        "course": "Cohort Analysis in Marketing",
        "link": "https://www.coursera.org/learn/cohort-analysis-marketing"
    },
    "CI/CD Pipeline Monitoring": {
        "course": "CI/CD Pipeline Monitoring and Maintenance",
        "link": "https://www.udemy.com/course/cicd-pipeline-monitoring-and-maintenance/"
    },
    "Typography Animation": {
        "course": "Typography Animation in After Effects",
        "link": "https://www.udemy.com/course/typography-animation-in-after-effects/"
    },
    "Service Level Agreement (SLA) Management": {
        "course": "SLA Management and Optimization",
        "link": "https://www.coursera.org/learn/sla-management"
    },
    "Analytical Reasoning": {
        "course": "Analytical Reasoning and Critical Thinking",
        "link": "https://www.udemy.com/course/analytical-reasoning-and-critical-thinking/"
    },
    "Motion Graphics": {
        "course": "Motion Graphics Masterclass with After Effects",
        "link": "https://www.udemy.com/course/motion-graphics-masterclass-with-after-effects/"
    },
    "Market Research": {
        "course": "Market Research and Consumer Behavior",
        "link": "https://www.coursera.org/learn/market-research"
    },
    "Agile Retrospectives": {
        "course": "Agile Retrospectives for Continuous Improvement",
        "link": "https://www.udemy.com/course/agile-retrospectives-for-continuous-improvement/"
    },
    "Networking": {
        "course": "Networking Fundamentals",
        "link": "https://www.coursera.org/learn/networking"
    },
    "Experiment Design": {
        "course": "Experiment Design for Research and Data Science",
        "link": "https://www.coursera.org/learn/experiment-design"
    },
    "Classroom Management": {
        "course": "Classroom Management Strategies",
        "link": "https://www.udemy.com/course/classroom-management-strategies/"
    },
    "Creative Play Facilitation": {
        "course": "Facilitating Creative Play for Children",
        "link": "https://www.udemy.com/course/facilitating-creative-play-for-children/"
    },
    "Crime Scene Documentation": {
        "course": "Crime Scene Documentation and Photography",
        "link": "https://www.udemy.com/course/crime-scene-documentation-and-photography/"
    },
    "Emergency Response Coordination": {
        "course": "Emergency Response Coordination and Management",
        "link": "https://www.udemy.com/course/emergency-response-coordination-and-management/"
    },
    "Skin Treatment Planning": {
        "course": "Skin Treatment Planning for Dermatologists",
        "link": "https://www.udemy.com/course/skin-treatment-planning-for-dermatologists/"
    },
    "Online Teaching Tools": {
        "course": "Effective Online Teaching Tools and Strategies",
        "link": "https://www.udemy.com/course/effective-online-teaching-tools-and-strategies/"
    },
    "Seasonal Decoration Design": {
        "course": "Seasonal Decoration and Floral Design",
        "link": "https://www.udemy.com/course/seasonal-decoration-and-floral-design/"
    },
    "Legal Procedure Knowledge": {
        "course": "Understanding Legal Procedures",
        "link": "https://www.udemy.com/course/understanding-legal-procedures/"
    },
    "Educational Psychology": {
        "course": "Educational Psychology for Teachers",
        "link": "https://www.coursera.org/learn/educational-psychology"
    },
    "Floral Supply Chain Management": {
        "course": "Floral Supply Chain Management",
        "link": "https://www.udemy.com/course/floral-supply-chain-management/"
    },
    "Report Writing (Law Enforcement)": {
        "course": "Report Writing for Law Enforcement Officers",
        "link": "https://www.udemy.com/course/report-writing-for-law-enforcement-officers/"
    },
    "Cosmetic Dermatology Procedures": {
        "course": "Cosmetic Dermatology Procedures for Beginners",
        "link": "https://www.udemy.com/course/cosmetic-dermatology-procedures-for-beginners/"
    },
    "Color Coordination (Floral Design)": {
        "course": "Floral Design and Color Coordination",
        "link": "https://www.udemy.com/course/floral-design-and-color-coordination/"
    },
    "Special Needs Instruction": {
        "course": "Special Needs Education and Instruction",
        "link": "https://www.udemy.com/course/special-needs-education-and-instruction/"
    },
    "Sympathy Flower Arrangement": {
        "course": "Sympathy and Funeral Flower Arrangement",
        "link": "https://www.udemy.com/course/sympathy-and-funeral-flower-arrangement/"
    },
    "Academic Advising": {
        "course": "Academic Advising for College Students",
        "link": "https://www.udemy.com/course/academic-advising-for-college-students/"
    },
    "Night Surveillance Protocol": {
        "course": "Night Surveillance Protocols for Security",
        "link": "https://www.udemy.com/course/night-surveillance-protocols-for-security/"
    },
    "Pediatric Skin Care": {
        "course": "Pediatric Skin Care and Treatment",
        "link": "https://www.udemy.com/course/pediatric-skin-care-and-treatment/"
    },
    "Electrical Circuit Design": {
        "course": "Electrical Circuit Design for Beginners",
        "link": "https://www.udemy.com/course/electrical-circuit-design-for-beginners/"
    },
}

def get_learning_resources(skills):
    """
    Fetch learning resources for given skills.
    Returns a dictionary mapping each skill to a list of resources.
    """
    recommendations = {}

    for skill in skills:
        skill_name = skill.get('skill_name') if isinstance(skill, dict) else skill

        # Fetch resources from the LEARNING_RESOURCES dictionary
        if skill_name in LEARNING_RESOURCES:
            recommendations[skill_name] = LEARNING_RESOURCES[skill_name]
        else:
            recommendations[skill_name] = ['No resources available for this skill.']

    return recommendations


def get_learning_resources_for_missing_and_improving_skills(missing_skills, skills_to_improve):
    """
    Fetch learning resources for both missing skills and skills to improve.
    """
    # Fetch learning resources for missing skills
    missing_resources = get_learning_resources(missing_skills)
    
    # Fetch learning resources for skills to improve
    improving_resources = get_learning_resources(skills_to_improve)
    
    # Combine both sets of recommendations
    all_resources = {}

    # Combine the dictionaries, ensuring no skill is overwritten
    for skill, resources in missing_resources.items():
        if skill not in all_resources:
            all_resources[skill] = resources
        else:
            all_resources[skill].extend(resources)

    for skill, resources in improving_resources.items():
        if skill not in all_resources:
            all_resources[skill] = resources
        else:
            all_resources[skill].extend(resources)
    
    return all_resources
