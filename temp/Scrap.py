import requests
from bs4 import BeautifulSoup

# Fetch the webpage
url = "https://asl-lex.org/"
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    titles = [item.text for item in soup.find_all('h2')]
    print(titles)
else:
    print("Failed to access the website.")
