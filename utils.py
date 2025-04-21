# skill_assessment/utils.py

def generate_dynamic_learning_links(skill):
    query = skill.replace(" ", "+")
    return [
        {
            "site": "W3Schools",
            "url": f"https://www.w3schools.com/search/search_result.asp?cx=012971019331610648934%3Afdngp0dl4gq&q={query}"
        },
        {
            "site": "FreeCodeCamp",
            "url": f"https://www.google.com/search?q=site%3Afreecodecamp.org+{query}"
        },
        {
            "site": "Coursera",
            "url": f"https://www.coursera.org/search?query={query}"
        },
        {
            "site": "Khan Academy",
            "url": f"https://www.khanacademy.org/search?page_search_query={query}"
        },
        {
            "site": "edX",
            "url": f"https://www.edx.org/search?q={query}"
        },
        {
            "site": "YouTube",
            "url": f"https://www.youtube.com/results?search_query={query}+tutorial"
        }
    ]
