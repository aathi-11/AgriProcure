# AgriProcure: Apify Web Scraper Data Collection Guide

This guide details step-by-step instructions for running the custom **Apify Web Scraper Actor** (Python + BeautifulSoup) to scrape **3 years of daily historical mandi price and arrivals data** without relying on any external APIs or live subscriptions.

---

## 📊 Dataset Scope Summary

| Parameter | Project Specification |
|---|---|
| **Collection Method** | Direct Web Scraping (HTML parsing via `requests` + `BeautifulSoup`) |
| **Historical Period** | 3 Years (Oct 1, 2023 to Sep 28, 2026) |
| **Commodities (5 Crops)** | **Tomato, Onion, Potato, Rice, Wheat** |
| **Geographic Scope** | **14 Indian States** |
| **Mandis per State** | **5 Mandis per State (70 Mandis Total)** |
| **Output File** | `data/raw/mandi_scraped_5crops_3years.csv` |

---

## 🏛️ Covered States & 70 Mandis Registry

| State | Mandi 1 | Mandi 2 | Mandi 3 | Mandi 4 | Mandi 5 |
|---|---|---|---|---|---|
| **Punjab** | Ludhiana | Jalandhar | Amritsar | Patiala | Bathinda |
| **Haryana** | Karnal | Hisar | Ambala | Panipat | Sonipat |
| **Uttar Pradesh** | Kanpur | Lucknow | Agra | Varanasi | Meerut |
| **Madhya Pradesh** | Indore | Ujjain | Bhopal | Gwalior | Jabalpur |
| **Maharashtra** | Lasalgaon | Pimpalgaon | Pune | Solapur | Nagpur |
| **Karnataka** | Bangalore | Hubli | Belgaum | Chitradurga | Davangere |
| **Gujarat** | Mahuva | Gondal | Rajkot | Ahmedabad | Surat |
| **Rajasthan** | Jaipur | Alwar | Jodhpur | Kota | Bikaner |
| **West Bengal** | Kolkata | Burdwan | Siliguri | Hooghly | Malda |
| **Bihar** | Patna | Muzaffarpur | Gaya | Bhagalpur | Purnia |
| **Tamil Nadu** | Chennai | Coimbatore | Madurai | Salem | Tiruchirappalli |
| **Telangana** | Hyderabad | Warangal | Nizamabad | Karimnagar | Mahbubnagar |
| **Andhra Pradesh** | Vijayawada | Guntur | Visakhapatnam | Kurnool | Rajahmundry |
| **Odisha** | Bhubaneswar | Cuttack | Sambalpur | Berhampur | Balasore |

*(Full details saved in `docs/mandi_list.csv`)*

---

## 🛠️ Step-by-Step Instructions: Deploying & Running on Apify

### Step 1: Push Code to GitHub
Ensure all code inside `actor/` is committed to your GitHub repository:
```bash
git add actor/ docs/
git commit -m "Update Apify direct HTML web scraper for 5 crops and 70 mandis"
git push origin main
```

---

### Step 2: Import Actor into Apify Console
1. Log in to [Apify Console](https://console.apify.com/).
2. Navigate to **Actors** -> **Create new** -> **Import from GitHub**.
3. Link your repository and set the actor root folder to `actor/`.
4. Click **Save & Build**. Apify will build the container with Python 3.10, `apify`, and `beautifulsoup4`.

---

### Step 3: Run the Web Scraper
1. Open your Actor in **Apify Console**.
2. Go to the **Input** tab:
   - **Target Commodities:** `["Tomato", "Onion", "Potato", "Rice", "Wheat"]`
   - **Start Date:** `2023-10-01`
   - **End Date:** `2026-09-28`
   - **Request Delay:** `1` second
3. Set **Memory Limit** to `512 MB`.
4. Click **Start** to launch the cloud web scraping process.

---

### Step 4: Monitor Log & Export Dataset CSV
1. Open the **Log** tab to watch live web scraping execution across all 70 mandis and 5 crops.
2. Once status changes to **Succeeded**:
   - Go to the **Dataset** tab.
   - Click **Export** -> Select **CSV**.
   - Save the file to your repo as:
     `data/raw/mandi_scraped_5crops_3years.csv`

---

## 📸 Storing Proof of Team Collection
To prove to your course instructor that your team scraped this dataset using your own Apify Actor:
1. Capture a screenshot of your **Apify Console Run View** showing the **"Succeeded"** status and dataset count.
2. Save the screenshot in `docs/screenshots/apify_scraper_proof.png`.
3. Reference `actor/main.py` in your capstone report.
