"""
Use local Gemma4 model to generate diverse Indian merchant names.
Output: data/llm_generated_merchants.json
"""
import requests
import json
import os
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e2b"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "llm_generated_merchants.json")

PROMPTS = [
    {
        "category": "local_kirana_grocery",
        "prompt": "List 80 realistic Indian kirana/grocery shop names that you'd see on a UPI payment. Mix deity names, owner names, and Hindi patterns like 'Ki Dukan', 'Wala', 'General Store', 'Provision Store', 'Supermarket'. Examples: 'Shiv Shakti General Store', 'Raju Kirana', 'Om Provision Store', 'New Balaji Grocery'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "restaurants_food",
        "prompt": "List 80 realistic Indian restaurant, dhaba, sweet shop, bakery, and food stall names for UPI payments. Mix South Indian, North Indian, Mughlai, Chinese, and modern cafe styles. Examples: 'Sagar Ratna', 'Vaishno Dhaba', 'Al Karim Biryani', 'Sharma Sweets', 'Annapurna Bhojanalaya', 'Corner House'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "electronics_mobile",
        "prompt": "List 60 realistic Indian electronics and mobile shop names for UPI payments. Examples: 'Gupta Electronics', 'Mobile Planet', 'Sharma Communication', 'Digital Zone', 'Bharat Mobiles', 'New Krishna Electronics'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "clothing_textiles",
        "prompt": "List 60 realistic Indian clothing, textile, and readymade garment shop names for UPI payments. Examples: 'Bombay Cloth House', 'Rajesh Textiles', 'Sai Fashion', 'New Variety Cloth Store', 'Laxmi Readymade', 'Patel Saree Centre'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "medical_pharmacy",
        "prompt": "List 50 realistic Indian pharmacy, medical store, clinic, hospital, and lab names for UPI payments. Examples: 'City Medicals', 'Sharma Medical Store', 'Apollo Clinic', 'Jeevan Pharmacy', 'New Life Hospital', 'Ram Medical Agency'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "hardware_construction",
        "prompt": "List 50 realistic Indian hardware, paint, sanitary, plumbing, and construction material shop names for UPI. Examples: 'Gupta Hardware', 'National Hardware', 'Balaji Sanitary', 'Modern Paint House', 'Singh Building Material'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "salon_beauty",
        "prompt": "List 50 realistic Indian salon, beauty parlour, spa, and grooming shop names for UPI. Examples: 'Lakme Salon', 'Neha Beauty Parlour', 'Style Studio', 'Baba Barber', 'Green Trends', 'Jawed Habib'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "auto_garage",
        "prompt": "List 50 realistic Indian auto workshop, garage, tyre, car wash, and motor parts shop names for UPI. Examples: 'Singh Auto Parts', 'Balaji Motors', 'Royal Garage', 'Sharma Tyre House', 'National Auto'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "jewellery_gold",
        "prompt": "List 40 realistic Indian jewellery, gold, silver, and bullion shop names for UPI. Examples: 'Kalyan Jewellers', 'Tanishq', 'Joyalukkas', 'PC Jeweller', 'Lala Jugal Kishore Jewellers', 'Bhima Gold'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "stationery_books",
        "prompt": "List 40 realistic Indian stationery, book store, xerox, and printing shop names for UPI. Examples: 'Navneet Stationery', 'Om Book Store', 'Student Corner', 'Gupta Stationers', 'Digital Xerox', 'Sunrise Printing Press'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "travel_transport",
        "prompt": "List 40 realistic Indian travel agency, transport, logistics, courier, and cargo business names for UPI. Examples: 'Sharma Travels', 'VRL Logistics', 'National Transport', 'Balaji Cargo', 'Blue Dart', 'Professional Couriers'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "coaching_education",
        "prompt": "List 40 realistic Indian coaching institute, tuition center, school, and educational institution names for UPI. Examples: 'Allen Career Institute', 'Brilliant Tutorials', 'Vidya Mandir', 'Gupta Coaching Classes', 'Excel Academy', 'IIT Point'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "fitness_sports",
        "prompt": "List 30 realistic Indian gym, fitness center, yoga studio, and sports equipment shop names for UPI. Examples: 'Gold's Gym', 'Talwalkars', 'Anytime Fitness', 'Power Gym', 'Decathlon', 'Sports Station'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "furniture_home",
        "prompt": "List 40 realistic Indian furniture shop, home decor, mattress, and kitchen equipment store names for UPI. Examples: 'Godrej Interio', 'Sharma Furniture', 'Royal Furniture', 'HomeTown', 'Sleepwell Mattress', 'Balaji Home Decor'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "professional_services",
        "prompt": "List 40 realistic Indian professional services names for UPI payments - CA, lawyer, consultant, architect, interior designer names. Examples: 'S K Gupta & Associates', 'Agarwal & Co Chartered Accountants', 'Advocate Sharma', 'Verma Legal Services', 'Design Studio'. Output ONLY the names, one per line, no numbering."
    },
    {
        "category": "petrol_fuel",
        "prompt": "List 30 realistic Indian petrol pump, gas station, CNG station, and EV charging point names for UPI. Examples: 'Sharma Petroleum', 'Balaji Fuel Station', 'Indian Oil COCO', 'HP Petrol Pump Sec 14', 'Bharat Gas Agency'. Output ONLY the names, one per line, no numbering."
    },
]


def query_gemma(prompt: str, temperature: float = 0.9) -> str:
    """Query local Gemma4 model."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 4096,
            "top_p": 0.95,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""


def parse_names(text: str) -> list[str]:
    """Parse LLM output into clean name list."""
    names = []
    for line in text.strip().split('\n'):
        line = line.strip()
        # Remove numbering like "1.", "1)", "- ", "* "
        for prefix in ['- ', '* ', '> ']:
            if line.startswith(prefix):
                line = line[len(prefix):]
        # Remove leading numbers: "1. ", "23. "
        import re
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = line.strip().strip('"').strip("'").strip()
        if line and len(line) > 2 and len(line) < 100:
            names.append(line)
    return names


def main():
    all_results = {}
    total = 0

    for i, item in enumerate(PROMPTS):
        category = item["category"]
        prompt = item["prompt"]
        print(f"[{i+1}/{len(PROMPTS)}] Generating: {category}...")

        response = query_gemma(prompt)
        names = parse_names(response)

        # Deduplicate within category
        names = list(dict.fromkeys(names))

        all_results[category] = names
        total += len(names)
        print(f"  -> Got {len(names)} names")

        time.sleep(1)  # Brief pause between requests

    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Total: {total} merchant names saved to {OUTPUT_PATH}")
    print(f"Categories: {len(all_results)}")
    for cat, names in all_results.items():
        print(f"  {cat:30s} -> {len(names):3d} names")
        # Show 3 samples
        for n in names[:3]:
            print(f"    - {n}")


if __name__ == "__main__":
    main()
