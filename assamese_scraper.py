import requests
from bs4 import BeautifulSoup
import re

# --- Configuration ---
TARGET_URL = "https://as.wikipedia.org/wiki/অসম" 
OUTPUT_FILE = "assamese_text.txt"

# --- HEADERS (The Fix) ---
# We mimic a standard browser so the server thinks a human is visiting.
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'as,en-US;q=0.9,en;q=0.8'
}

def is_assamese(text, threshold=0.5):
    """
    Checks if text is Assamese. 
    Prioritizes strict character checks ('ৰ', 'ৱ') before falling back to Unicode block ratios.
    """
    if not text:
        return False
        
    # 1. STRICT CHECK: 'ৰ' (Ra) and 'ৱ' (Wa) are unique to Assamese
    if 'ৰ' in text or 'ৱ' in text:
        return True

    # 2. FALLBACK RATIO CHECK
    clean_text = re.sub(r'\s+', '', text)
    if len(clean_text) == 0:
        return False

    assamese_char_count = 0
    for char in clean_text:
        # Bengali-Assamese Unicode Block
        if '\u0980' <= char <= '\u09FF':
            assamese_char_count += 1
            
    ratio = assamese_char_count / len(clean_text)
    return ratio > threshold

def scrape_assamese_text(url):
    print(f"Scraping: {url}...")
    try:
        # --- THE FIX IS HERE ---
        # We pass the 'headers' dictionary to the get request
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from standard content tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'span'])
        
        collected_lines = []
        for element in text_elements:
            raw_text = element.get_text(strip=True)
            
            if is_assamese(raw_text):
                collected_lines.append(raw_text)

        return collected_lines

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def save_to_file(lines, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Successfully saved {len(lines)} lines to {filename}")

if __name__ == "__main__":
    data = scrape_assamese_text(TARGET_URL)
    if data:
        save_to_file(data, OUTPUT_FILE)
        print("\n--- Preview ---")
        for line in data[:3]:
            print(line)
    else:
        print("No Assamese text found.")
