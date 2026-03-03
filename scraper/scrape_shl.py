# ============================================================
# SHL Product Catalog - Production-Grade Web Scraper
# Target : https://www.shl.com/products/product-catalog/
# Output : data/shl_assessments_final.csv
#          data/failed_links.csv
#          data/backup_<n>.csv  (every 20 products)
# ============================================================
#
# REAL SITE STRUCTURE (verified):
#   Catalog paginated at: /products/product-catalog/?start=N&type=1
#   Product pages at    : /products/product-catalog/view/<slug>/
#   Links are RELATIVE  : href="/products/product-catalog/view/..."
#   Table selector      : div.custom__table-responsive table tbody tr
#   Columns             : [name/link] [remote dot] [adaptive dot] [test-type badges]
#   Test-type badges    : <span class="test-type">A</span>  (letters A B C E K P S)
#   Green dot = Yes     : <span class="dot -green">
#
# TEST TYPE LETTER MAP:
#   A = Ability & Aptitude
#   B = Personality & Behavior
#   C = Competencies
#   E = Assessment Exercises
#   K = Knowledge & Skills
#   P = Personality & Behavior  (alternate)
#   S = Simulations
#   D = Development & 360
# ============================================================

import re
import os
import time
import traceback

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BASE_DOMAIN  = "https://www.shl.com"
CATALOG_PATH = "/products/product-catalog/"
CATALOG_URL  = BASE_DOMAIN + CATALOG_PATH

DATA_DIR     = "data"
FINAL_CSV    = os.path.join(DATA_DIR, "shl_assessments_final.csv")
FAILED_CSV   = os.path.join(DATA_DIR, "failed_links.csv")
BACKUP_EVERY = 20

PAGE_LOAD_WAIT = 15   # seconds
ELEMENT_WAIT   = 10   # seconds
POLITE_DELAY   = 1.5  # seconds between product page requests

# Test-type letter → human label
TEST_TYPE_LETTER_MAP = {
    "A": "Ability & Aptitude",
    "B": "Personality & Behavior",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# Fallback keyword map for product pages that don't have badges
FALLBACK_KEYWORDS = [
    ("assessment exercise",  "Assessment Exercises"),
    ("360",                  "Development & 360"),
    ("development",          "Development & 360"),
    ("simulation",           "Simulations"),
    ("competenc",            "Competencies"),
    ("personality",          "Personality & Behavior"),
    ("behavior",             "Personality & Behavior"),
    ("behaviour",            "Personality & Behavior"),
    ("knowledge",            "Knowledge & Skills"),
    ("skills",               "Knowledge & Skills"),
    ("ability",              "Ability & Aptitude"),
    ("aptitude",             "Ability & Aptitude"),
    ("cognitive",            "Ability & Aptitude"),
    ("verbal",               "Ability & Aptitude"),
    ("numerical",            "Ability & Aptitude"),
    ("inductive",            "Ability & Aptitude"),
    ("deductive",            "Ability & Aptitude"),
]


# ─────────────────────────────────────────────────────────────
# BROWSER SETUP
# ─────────────────────────────────────────────────────────────

def create_driver() -> webdriver.Chrome:
    """Launch Chrome (non-headless) with anti-bot settings."""
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


# ─────────────────────────────────────────────────────────────
# PHASE 1: COLLECT PRODUCT LINKS FROM CATALOG TABLE
# ─────────────────────────────────────────────────────────────

def _wait_for_table(driver: webdriver.Chrome) -> bool:
    """Wait until the catalog table is present. Returns True if found."""
    try:
        WebDriverWait(driver, PAGE_LOAD_WAIT).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.custom__table-responsive table tbody tr")
            )
        )
        return True
    except TimeoutException:
        return False


def _extract_links_from_page(driver: webdriver.Chrome) -> list[str]:
    """
    Extract product URLs from the catalog table on the current page.
    Links are relative (/products/product-catalog/view/<slug>/).
    Returns a list of absolute URLs.
    """
    links = []
    try:
        rows = driver.find_elements(
            By.CSS_SELECTOR,
            "div.custom__table-responsive table tbody tr"
        )
        for row in rows:
            try:
                anchor = row.find_element(By.CSS_SELECTOR, "td:first-child a")
                href = anchor.get_attribute("href") or ""
                # Selenium returns absolute URLs when the driver is navigated
                if "/product-catalog/view/" in href:
                    links.append(href.rstrip("/"))
            except Exception:
                pass
    except Exception as e:
        print(f"  [WARN] Could not parse table rows: {e}")
    return links


def collect_product_links(driver: webdriver.Chrome) -> list[str]:
    """
    Paginate through the catalog using ?start=N&type=1 (Individual Tests only).
    Falls back to a no-filter sweep to catch any extras.
    """
    all_links: set[str] = set()

    def _paginate(url_template: str, label: str):
        start = 0
        step = 12
        consecutive_empty = 0
        MAX_EMPTY = 3

        print(f"\n[CATALOG] {label} …")

        while True:
            page_url = url_template.format(start=start)
            print(f"  → [{start:>4d}] {page_url}")

            try:
                driver.get(page_url)
            except WebDriverException as e:
                print(f"  [ERROR] Could not load page: {e}")
                consecutive_empty += 1
            else:
                found = _wait_for_table(driver)
                if found:
                    page_links = _extract_links_from_page(driver)
                    new = [l for l in page_links if l not in all_links]
                    if new:
                        all_links.update(new)
                        consecutive_empty = 0
                        print(f"       +{len(new)} new  | total {len(all_links)}")
                    else:
                        consecutive_empty += 1
                        print(f"       No new links ({consecutive_empty}/{MAX_EMPTY})")
                else:
                    consecutive_empty += 1
                    print(f"       Table not found ({consecutive_empty}/{MAX_EMPTY})")

            if consecutive_empty >= MAX_EMPTY:
                print("  [STOP] Pagination complete.")
                break

            start += step

    # Pass 1: type=1 filter (Individual Test Solutions)
    _paginate(CATALOG_URL + "?start={start}&type=1", "Pass 1 – Individual Tests (type=1)")

    # Pass 2: no filter sweep (wider net)
    _paginate(CATALOG_URL + "?start={start}", "Pass 2 – Full catalog (no filter)")

    final = sorted(all_links)
    print(f"\n[LINKS] Total unique product links: {len(final)}")
    return final


# ─────────────────────────────────────────────────────────────
# PHASE 2: EXTRACT DATA FROM EACH PRODUCT PAGE
# ─────────────────────────────────────────────────────────────

def extract_name(driver: webdriver.Chrome) -> str:
    try:
        h1 = WebDriverWait(driver, ELEMENT_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
        )
        return h1.text.strip()
    except TimeoutException:
        return "Not Available"


def extract_description(driver: webdriver.Chrome) -> str:
    """
    Return the longest meaningful paragraph found in the main content area.
    """
    selectors = ["main p", "article p", ".hero__text p", "section p", "p"]
    for sel in selectors:
        try:
            paras = driver.find_elements(By.CSS_SELECTOR, sel)
            candidates = [p.text.strip() for p in paras if len(p.text.strip()) > 40]
            if candidates:
                return max(candidates, key=len)
        except Exception:
            pass
    return "Not Available"


def extract_duration(page_source: str) -> str:
    patterns = [
        r"\d+\s*[-–to]+\s*\d+\s*min(?:utes?)?",
        r"approx(?:imately)?\s*\d+\s*min(?:utes?)?",
        r"\d+\s*min(?:utes?)?",
        r"\d+\s*hour(?:s)?",
    ]
    text = page_source.lower()
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(0).strip()
    return "Not Available"


def extract_adaptive_remote_from_catalog_row(row_html: str) -> tuple[str, str]:
    """
    Parse remote & adaptive from catalog table row HTML.
    A green dot (.dot.-green) in column 2 = Remote Yes.
    A green dot in column 3 = Adaptive Yes.
    """
    # We use simple string checks since we already have the row HTML
    cols = row_html.split("<td")
    remote   = "No"
    adaptive = "No"

    if len(cols) > 2 and "-green" in cols[2]:
        remote = "Yes"
    if len(cols) > 3 and "-green" in cols[3]:
        adaptive = "Yes"

    return remote, adaptive


def _get_dot_columns(driver: webdriver.Chrome, url: str, all_rows_data: dict) -> tuple[str, str]:
    """
    Look up pre-collected catalog row data for this URL's remote/adaptive dots.
    Falls back to page-text detection if row data isn't available.
    """
    if url in all_rows_data:
        r = all_rows_data[url]
        return r["remote_support"], r["adaptive_support"]
    return "Not Available", "Not Available"


def extract_test_type_from_page(driver: webdriver.Chrome, page_source: str) -> str:
    """
    Try to find test-type by:
    1. structured span.test-type badges on the product page itself
    2. keyword fallback
    """
    # 1) Look for span.test-type badges on the product page
    try:
        badges = driver.find_elements(By.CSS_SELECTOR, "span.product-catalogue__type, span[class*='type']")
        letters = [b.text.strip().upper() for b in badges if b.text.strip()]
        labels = [TEST_TYPE_LETTER_MAP[l] for l in letters if l in TEST_TYPE_LETTER_MAP]
        if labels:
            return " | ".join(dict.fromkeys(labels))  # unique, ordered
    except Exception:
        pass

    # 2) Keyword fallback
    text = page_source.lower()
    for keyword, label in FALLBACK_KEYWORDS:
        if keyword in text:
            return label

    return "Not Available"


def scrape_product_page(driver: webdriver.Chrome, url: str, row_data: dict) -> dict | None:
    """Scrape a single SHL product page. Returns dict or None on failure."""
    try:
        driver.get(url)

        try:
            WebDriverWait(driver, PAGE_LOAD_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
            )
        except TimeoutException:
            print(f"       [WARN] H1 timeout — continuing anyway")

        page_source = driver.page_source

        name        = extract_name(driver)
        description = extract_description(driver)
        duration    = extract_duration(page_source)
        test_type   = extract_test_type_from_page(driver, page_source)

        # Use catalog-row dot data if available, else page-text fallback
        remote, adaptive = _get_dot_columns(driver, url, row_data)
        if remote == "Not Available":
            remote   = "Yes" if "remote"   in page_source.lower() else "No"
        if adaptive == "Not Available":
            adaptive = "Yes" if "adaptive" in page_source.lower() else "No"

        return {
            "name":             name,
            "url":              url,
            "description":      description,
            "duration":         duration,
            "remote_support":   remote,
            "adaptive_support": adaptive,
            "test_type":        test_type,
        }

    except WebDriverException as e:
        print(f"       [ERROR] WebDriverException: {e}")
        return None
    except Exception as e:
        print(f"       [ERROR] Unexpected: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────
# COLLECT CATALOG ROW METADATA (remote/adaptive/test-type)
# ─────────────────────────────────────────────────────────────

def collect_catalog_row_metadata(driver: webdriver.Chrome) -> dict:
    """
    Re-paginate the catalog and extract per-row metadata:
    url → {remote_support, adaptive_support, test_type}
    This avoids having to parse these from product pages.
    """
    row_data: dict = {}

    def _parse_rows(url_template: str, label: str):
        start = 0
        step = 12
        consecutive_empty = 0
        MAX_EMPTY = 3

        print(f"\n[META] Collecting row metadata: {label} …")

        while True:
            page_url = url_template.format(start=start)
            try:
                driver.get(page_url)
                found = _wait_for_table(driver)
                if not found:
                    consecutive_empty += 1
                    if consecutive_empty >= MAX_EMPTY:
                        break
                    start += step
                    continue
            except Exception:
                consecutive_empty += 1
                if consecutive_empty >= MAX_EMPTY:
                    break
                start += step
                continue

            try:
                rows = driver.find_elements(
                    By.CSS_SELECTOR,
                    "div.custom__table-responsive table tbody tr"
                )
            except Exception:
                consecutive_empty += 1
                if consecutive_empty >= MAX_EMPTY:
                    break
                start += step
                continue

            new_found = 0
            for row in rows:
                try:
                    anchor = row.find_element(By.CSS_SELECTOR, "td:first-child a")
                    href = anchor.get_attribute("href") or ""
                    if "/product-catalog/view/" not in href:
                        continue
                    url = href.rstrip("/")

                    # Column 2: remote dot
                    try:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        remote   = "No"
                        adaptive = "No"
                        if len(cols) > 1:
                            dot_remote = cols[1].find_elements(By.CSS_SELECTOR, "span.-green, .dot.-green, [class*='green']")
                            remote = "Yes" if dot_remote else "No"
                        if len(cols) > 2:
                            dot_adaptive = cols[2].find_elements(By.CSS_SELECTOR, "span.-green, .dot.-green, [class*='green']")
                            adaptive = "Yes" if dot_adaptive else "No"

                        # Column 4: test-type badges
                        test_letters = []
                        if len(cols) > 3:
                            badges = cols[3].find_elements(By.CSS_SELECTOR, "span")
                            for badge in badges:
                                letter = badge.text.strip().upper()
                                if letter in TEST_TYPE_LETTER_MAP:
                                    lbl = TEST_TYPE_LETTER_MAP[letter]
                                    if lbl not in test_letters:
                                        test_letters.append(lbl)

                        test_type = " | ".join(test_letters) if test_letters else "Not Available"

                    except Exception:
                        remote, adaptive, test_type = "No", "No", "Not Available"

                    if url not in row_data:
                        row_data[url] = {
                            "remote_support":   remote,
                            "adaptive_support": adaptive,
                            "test_type":        test_type,
                        }
                        new_found += 1

                except Exception:
                    pass

            if new_found == 0:
                consecutive_empty += 1
                if consecutive_empty >= MAX_EMPTY:
                    break
            else:
                consecutive_empty = 0
                print(f"  [{start:>4d}] +{new_found} rows | total {len(row_data)}")

            start += step

    _parse_rows(CATALOG_URL + "?start={start}&type=1", "Individual Tests (type=1)")
    _parse_rows(CATALOG_URL + "?start={start}",        "Full catalog (no filter)")

    print(f"\n[META] Collected row metadata for {len(row_data)} products")
    return row_data


# ─────────────────────────────────────────────────────────────
# SAVE UTILITIES
# ─────────────────────────────────────────────────────────────

def save_backup(records: list[dict], n: int) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"backup_{n}.csv")
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"  [BACKUP] {len(records)} records → {path}")


def save_final(records: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame(records)
    before = len(df)
    df.drop_duplicates(subset=["url"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(FINAL_CSV, index=False)
    print(f"  [SAVED] {len(df)} records (removed {before - len(df)} dupes) → {FINAL_CSV}")


def save_failed(failed: list[dict]) -> None:
    if not failed:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(failed).to_csv(FAILED_CSV, index=False)
    print(f"  [FAILED] {len(failed)} URLs → {FAILED_CSV}")


# ─────────────────────────────────────────────────────────────
# STRATEGY: BUILD FROM CATALOG TABLE (no product-page visits needed
#           for remote/adaptive/test_type — use row metadata directly)
# ─────────────────────────────────────────────────────────────

def build_dataset_from_catalog(driver: webdriver.Chrome) -> list[dict]:
    """
    Build a dataset entirely from catalog table metadata.
    Only visits product pages to get name + description + duration.
    """
    row_data = collect_catalog_row_metadata(driver)

    if not row_data:
        print("[ERROR] No catalog rows found. Check selectors or network.")
        return []

    urls = sorted(row_data.keys())
    total = len(urls)

    # Save raw links
    raw_path = os.path.join(DATA_DIR, "data_product_links.csv")
    pd.DataFrame(urls, columns=["url"]).to_csv(raw_path, index=False)
    print(f"\n[LINKS] Saved {total} raw product links → {raw_path}")

    records: list[dict] = []
    failed:  list[dict] = []

    print(f"\n[SCRAPE] Visiting {total} product pages …\n")

    for idx, url in enumerate(urls, start=1):
        meta = row_data[url]
        print(f"[{idx:4d}/{total}] {url}")

        result = scrape_product_page(driver, url, row_data)

        if result:
            # Override remote/adaptive/test_type with catalog-row values (more reliable)
            result["remote_support"]   = meta["remote_support"]
            result["adaptive_support"] = meta["adaptive_support"]
            if meta["test_type"] != "Not Available":
                result["test_type"] = meta["test_type"]
            records.append(result)
            print(f"       ✔  {result['name'][:65]}")
        else:
            failed.append({"url": url, "reason": "Scraping failed"})
            print(f"       ✗  FAILED")

        if len(records) > 0 and len(records) % BACKUP_EVERY == 0:
            save_backup(records, len(records))

        time.sleep(POLITE_DELAY)

    return records, failed


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_scraper() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    driver = create_driver()

    try:
        records, failed = build_dataset_from_catalog(driver)

        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE")
        print(f"  Successful : {len(records)}")
        print(f"  Failed     : {len(failed)}")
        print("=" * 60 + "\n")

        save_final(records)
        save_failed(failed)

    finally:
        driver.quit()
        print("[DONE] Browser closed.")


if __name__ == "__main__":
    run_scraper()
