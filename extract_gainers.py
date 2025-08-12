import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

from utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def suppress_stderr():
    """Redirect ``sys.stderr`` and ``sys.stdout`` to devnull within the context."""
    original_stderr, original_stdout = sys.stderr, sys.stdout
    # Explicit UTF-8 encoding ensures writing arbitrary Unicode characters
    # (including emojis) to the suppressed streams does not raise
    # ``UnicodeEncodeError`` on platforms where the default encoding is not
    # UTF-8 (e.g. Windows).  Errors are ignored so any problematic characters
    # are simply dropped rather than crashing the program.
    with open(os.devnull, "w", encoding="utf-8", errors="ignore") as devnull:
        with redirect_stderr(devnull), redirect_stdout(devnull):
            yield
    sys.stderr = original_stderr
    sys.stdout = original_stdout

def extract_gainers():
    """Scrape the CoinMarketCap gainers page.

    All noisy stderr output is suppressed during scraping and restored
    afterwards to avoid polluting callers' logs.
    """

    url = "https://coinmarketcap.com/gainers-losers/"
    service = Service(ChromeDriverManager().install())
    service.log_path = os.devnull  # suppress ChromeDriver logs cross-platform

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # ✅ Headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36",
    )

    tf_log = os.environ.get("TF_CPP_MIN_LOG_LEVEL")
    absl_log = os.environ.get("ABSL_LOG_LEVEL")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_LOG_LEVEL"] = "FATAL"

    driver = None

    try:
        for attempt in range(2):
            try:
                with suppress_stderr():
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    driver.get(url)
                break
            except WebDriverException as e:
                if attempt == 0:
                    logger.warning(
                        "Chrome WebDriver failed to start (attempt %d). Retrying...", attempt + 1
                    )
                    time.sleep(1)
                else:
                    logger.error(
                        "❌ Unable to start Chrome WebDriver: %s. Ensure Google Chrome or Chromium is installed.",
                        e,
                    )
                    return []

        logger.info("⏳ Waiting for page to fully load...")
        time.sleep(5)

        tables = driver.find_elements(By.TAG_NAME, "table")
        if not tables:
            logger.warning("❌ No tables found.")
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
                symbol = (
                    name_symbol_block[1].text.strip()
                    if len(name_symbol_block) > 1
                    else ""
                )
                price = cols[2].text.strip()
                change = cols[3].text.strip()
                volume = cols[4].text.strip()

                gainers.append(
                    {
                        "name": name,
                        "symbol": symbol,
                        "price": price,
                        "change": change,
                        "volume": volume,
                    }
                )

            except Exception:
                continue

        return gainers

    except WebDriverException as e:
        logger.error(
            "❌ Unable to start Chrome WebDriver: %s. Ensure Google Chrome or Chromium is installed.",
            e,
        )
        return []
    finally:
        if driver:
            driver.quit()
        if tf_log is None:
            os.environ.pop("TF_CPP_MIN_LOG_LEVEL", None)
        else:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_log
        if absl_log is None:
            os.environ.pop("ABSL_LOG_LEVEL", None)
        else:
            os.environ["ABSL_LOG_LEVEL"] = absl_log

if __name__ == "__main__":
    gainers = extract_gainers()

    logger.info("\n📈 Top 10 Gainers:")
    for g in gainers[:10]:   # ✅ Limit to top 10
        logger.info("%s", g)
