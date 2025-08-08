import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# ✅ Suppress TensorFlow/C++/native stderr noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = 'FATAL'
sys.stderr = open(os.devnull, 'w')
os.dup2(sys.stderr.fileno(), 2)

def extract_gainers():
    url = "https://coinmarketcap.com/gainers-losers/"

    service = Service(ChromeDriverManager().install())
    service.log_path = "NUL"  # suppress ChromeDriver logs (Windows)

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # ✅ Headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)

    try:
        print("⏳ Waiting for page to fully load...")
        time.sleep(5)

        tables = driver.find_elements(By.TAG_NAME, "table")
        if not tables:
            print("❌ No tables found.")
            return []

        gainers = []
        rows = tables[0].find_elements(By.XPATH, ".//tbody/tr")

        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 5:
                continue

            try:
                name_symbol_block = cols[1].find_elements(By.TAG_NAME, "p")
                name = name_symbol_block[0].text.strip()
                symbol = name_symbol_block[1].text.strip() if len(name_symbol_block) > 1 else ""
                price = cols[2].text.strip()
                change = cols[3].text.strip()
                volume = cols[4].text.strip()

                gainers.append({
                    "name": name,
                    "symbol": symbol,
                    "price": price,
                    "change": change,
                    "volume": volume
                })

            except Exception:
                continue

        return gainers

    finally:
        driver.quit()

if __name__ == "__main__":
    gainers = extract_gainers()

    print("\n📈 Top 10 Gainers:")
    for g in gainers[:10]:   # ✅ Limit to top 10
        print(g)
