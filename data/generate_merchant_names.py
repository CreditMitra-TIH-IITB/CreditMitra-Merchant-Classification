"""
Indian Merchant Names Dataset Generator
=========================================
Generates a comprehensive dataset of merchant/business names for the Merchant Classifier.

Layers:
1. Real Indian & international brands (hardcoded)
2. Local business name patterns (combinatorial generation)
3. Legal entity names
4. Person-named businesses (ambiguous edge cases)
5. Synthetic/modern brand-style names
6. LLM-generated names (merged from llm_generated_merchants.json)
7. Augmented variants (noise, truncation, typos)

Output: data/indian_merchant_names.csv
"""

import csv
import random
import os
import json
import string


# ============================================================================
# LAYER 1: REAL BRANDS & COMPANIES
# ============================================================================

REAL_BRANDS = {
    # ---- E-COMMERCE ----
    "ecommerce": [
        "Amazon", "Amazon India", "Amazon Pay", "Flipkart", "Flipkart Wholesale",
        "Myntra", "Meesho", "Ajio", "Nykaa", "Nykaa Fashion", "FirstCry",
        "Snapdeal", "ShopClues", "Udaan", "IndiaMART", "TradeIndia",
        "Tata Cliq", "Tata Cliq Luxury", "JioMart", "Purplle", "Sugar Cosmetics",
        "Mamaearth", "BoAt", "Noise", "Fire-Boltt", "Croma",
        "Shopsy", "GlowRoad", "DealShare", "CityMall", "Roposo Clout",
        "LimeRoad", "Koovs", "Bewakoof", "The Souled Store", "Chumbak",
        "FabAlley", "StalkBuyLove", "Urbanic", "Clovia", "Zivame",
        "Pepperfry", "Urban Ladder", "WoodenStreet", "Hometown",
        "1mg", "Tata 1mg", "BigSmall",
    ],

    # ---- FOOD DELIVERY / QUICK COMMERCE ----
    "food_delivery": [
        "Swiggy", "Swiggy Instamart", "Swiggy Genie", "Zomato", "Zomato Gold",
        "Blinkit", "Zepto", "Dunzo", "BigBasket", "BigBasket Now",
        "JioMart", "Amazon Fresh", "Flipkart Quick", "Licious", "FreshToHome",
        "Country Delight", "Milkbasket", "Supr Daily", "BBDaily",
        "EatFit", "FreshMenu", "Box8", "Faasos", "Behrouz Biryani",
        "Rebel Foods", "Oven Story Pizza", "The Good Bowl", "Mandarin Oak",
        "iD Fresh Food", "Biryani By Kilo", "EatSure", "Dineout",
        "Eazydiner", "MagicPin", "CureFit Eat", "HungerBox",
        "Dominos", "McDonalds", "KFC", "Pizza Hut", "Burger King",
        "Subway", "Starbucks", "Costa Coffee",
    ],

    # ---- FINTECH / PAYMENTS ----
    "fintech": [
        "PhonePe", "PhonePe Merchant", "Paytm", "Paytm Mall", "Paytm Payments",
        "Google Pay", "GPay", "CRED", "CRED Mint", "Razorpay",
        "BharatPe", "MobiKwik", "FreeCharge", "PayU", "PayU India",
        "Cashfree", "Instamojo", "Simpl", "LazyPay", "ZestMoney",
        "Slice", "Jupiter", "Fi Money", "NiYO", "Uni Cards",
        "KreditBee", "MoneyTap", "Stashfin", "Dhani", "Bajaj Finserv",
        "Bajaj Finance", "Muthoot Finance", "Manappuram Finance",
        "Khatabook", "Vyapar", "myBillBook", "OkCredit", "Dukaan",
        "Groww", "Zerodha", "Angel One", "Upstox", "5Paisa",
        "Kuvera", "Coin by Zerodha", "Smallcase", "INDmoney", "Paisa Bazaar",
        "Policybazaar", "Digit Insurance", "Acko Insurance",
        "RazorpayX", "PayTM for Business", "Pine Labs",
    ],

    # ---- RIDE / MOBILITY ----
    "travel_mobility": [
        "Uber", "Uber India", "Uber Auto", "Uber Moto", "Ola", "Ola Cabs",
        "Ola Auto", "Rapido", "Rapido Bike", "BluSmart", "Yulu", "Bounce",
        "Meru Cabs", "Jugnoo", "InDrive", "Namma Yatri",
        "MakeMyTrip", "Goibibo", "Ixigo", "Cleartrip", "Yatra", "EaseMyTrip",
        "RedBus", "AbhiBus", "ConfirmTkt", "RailYatri", "IRCTC",
        "IndiGo", "Air India", "SpiceJet", "Vistara", "AirAsia India",
        "Akasa Air", "Go First", "Jet Airways",
        "OYO", "OYO Rooms", "Treebo", "FabHotels", "Zostel", "Hostelworld",
        "Taj Hotels", "Oberoi Hotels", "ITC Hotels", "Lemon Tree Hotels",
        "Club Mahindra", "Sterling Holidays", "Thomas Cook India",
    ],

    # ---- GROCERY / RETAIL CHAINS ----
    "grocery_retail": [
        "DMart", "D-Mart", "Avenue Supermarts", "Reliance Smart", "Reliance Fresh",
        "Reliance Smart Bazaar", "Reliance Retail", "Reliance Digital",
        "Reliance Trends", "Reliance Jewels", "AJIO Luxe",
        "More Supermarket", "More Megastore", "Star Bazaar", "Spencer's",
        "Spencer's Retail", "Nature's Basket", "Easyday", "Heritage Fresh",
        "Ratnadeep", "Nilgiris", "Foodhall",
        "Vishal Mega Mart", "V-Mart", "V2 Retail", "Metro Cash & Carry",
        "Walmart India", "Spar Hypermarket", "HyperCITY",
        "24Seven", "7-Eleven India",
    ],

    # ---- TELECOM / DTH / BROADBAND ----
    "telecom": [
        "Jio", "Reliance Jio", "JioFiber", "Jio Recharge",
        "Airtel", "Bharti Airtel", "Airtel Payments Bank", "Airtel Xstream",
        "Vi", "Vodafone Idea", "Vodafone", "Idea",
        "BSNL", "MTNL", "Tata Teleservices",
        "Tata Play", "Dish TV", "D2h", "Airtel Digital TV", "Sun Direct",
        "Hathway", "Den Networks", "ACT Fibernet", "Excitel", "Tikona",
    ],

    # ---- FOOD CHAINS / QSR / RESTAURANTS (National) ----
    "restaurant": [
        "Dominos Pizza", "McDonalds India", "KFC India", "Pizza Hut India",
        "Burger King India", "Subway India", "Starbucks India",
        "Haldirams", "Haldiram's", "Bikanervala", "Bikaner Sweets",
        "Sagar Ratna", "Saravana Bhavan", "Vaishno Devi Dhaba",
        "CCD", "Cafe Coffee Day", "Chaayos", "Chai Point", "Chai Sutta Bar",
        "Third Wave Coffee", "Blue Tokai Coffee", "Sleepy Owl",
        "Barbeque Nation", "Absolute Barbecues", "Mainland China",
        "The Yellow Chilli", "Pind Balluchi", "Moti Mahal",
        "Paradise Biryani", "Behrouz Biryani", "Biryani Blues",
        "Wow Momo", "Momos King", "Goli Vada Pav", "Jumboking",
        "Natural Ice Cream", "Baskin Robbins", "Kwality Walls",
        "Amul Ice Cream", "Havmor", "Cream Stone",
        "Keventers", "Shake Factory", "Belgian Waffle",
        "Theobroma", "Monginis", "Karachi Bakery", "Iyengar Bakery",
        "Wenger's", "Flurys", "Mrs Magpie",
        "Smoke House Deli", "Social", "Toit", "Brewbot",
        "Punjab Grill", "Dhaba by Claridges", "Indian Accent",
        "Bukhara", "Dum Pukht", "Karim's", "Al Jawahar",
    ],

    # ---- FASHION / LIFESTYLE ----
    "fashion": [
        "Zara", "Zara India", "H&M", "H&M India", "Uniqlo",
        "Levi's", "Levis India", "FabIndia", "Fabindia", "Westside",
        "Pantaloons", "Max Fashion", "Lifestyle", "Shoppers Stop",
        "Allen Solly", "Van Heusen", "Peter England", "Louis Philippe",
        "Raymond", "Raymond Shop", "Arvind", "Manyavar",
        "Titan", "Titan Eye Plus", "Tanishq", "CaratLane", "Mia by Tanishq",
        "Kalyan Jewellers", "Joyalukkas", "PC Jeweller", "Senco Gold",
        "Malabar Gold", "GRT Jewellers", "Bhima Jewellers",
        "Bata", "Bata India", "Metro Shoes", "Khadim's", "Liberty Shoes",
        "Woodland", "Puma India", "Nike India", "Adidas India",
        "Decathlon", "Decathlon India", "Skechers India",
        "W", "Aurelia", "Global Desi", "AND India",
        "Lenskart", "Lenskart Gold", "John Jacobs",
        "Titan World", "Fastrack", "Sonata",
    ],

    # ---- HEALTHCARE / PHARMACY ----
    "healthcare": [
        "Apollo Pharmacy", "Apollo Hospital", "Apollo Clinic",
        "Netmeds", "PharmEasy", "1mg", "Tata 1mg", "MedPlus", "MedPlus Pharmacy",
        "Practo", "Pristyn Care", "MFine", "DocsApp",
        "Max Hospital", "Max Healthcare", "Fortis Hospital", "Fortis Healthcare",
        "Narayana Health", "Manipal Hospital", "Columbia Asia",
        "AIIMS", "Medanta", "Kokilaben Hospital",
        "Dr Lal Path Labs", "SRL Diagnostics", "Thyrocare", "Metropolis Lab",
        "Portea Medical", "Cult Fit", "CureFit",
        "HealthKart", "Kapiva", "Himalaya Wellness",
    ],

    # ---- INSURANCE ----
    "insurance": [
        "LIC", "LIC of India", "Life Insurance Corporation",
        "HDFC Life", "HDFC Ergo", "ICICI Prudential", "ICICI Lombard",
        "SBI Life", "Max Life Insurance", "Bajaj Allianz",
        "Star Health Insurance", "Tata AIA", "Aditya Birla Sun Life",
        "Kotak Life Insurance", "PNB MetLife", "Bharti AXA",
        "New India Assurance", "United India Insurance",
        "National Insurance", "Oriental Insurance", "GIC Re",
        "Policybazaar", "Digit Insurance", "Acko Insurance",
        "Go Digit", "Niva Bupa Health",
    ],

    # ---- ED-TECH ----
    "education": [
        "BYJU'S", "Byjus", "Unacademy", "PhysicsWallah", "PW",
        "Vedantu", "Toppr", "Simplilearn", "UpGrad", "Coursera India",
        "Scaler", "Coding Ninjas", "GeeksforGeeks", "LeetCode",
        "Testbook", "Oliveboard", "Gradeup", "Adda247",
        "Aakash Institute", "Allen Career Institute", "FIITJEE",
        "Resonance", "Narayana", "Sri Chaitanya", "Career Point",
        "Embibe", "Doubtnut", "Extramarks",
        "WhiteHat Jr", "Cuemath", "LEAD School",
    ],

    # ---- UTILITIES ----
    "utility": [
        "BESCOM", "BSES", "BSES Rajdhani", "BSES Yamuna",
        "Tata Power", "Tata Power DDL", "Adani Electricity",
        "MSEDCL", "KSEB", "TNEB", "WBSEDCL", "CESC",
        "UPPCL", "UHBVN", "DHBVN", "JVVNL", "AVVNL",
        "Adani Gas", "Mahanagar Gas", "MGL", "IGL", "Gujarat Gas",
        "Indraprastha Gas", "Adani Total Gas",
        "MTNL", "BSNL Broadband", "Hathway Cable",
        "Municipal Corporation", "MCD", "BMC", "GHMC", "BBMP",
        "Water Board", "Delhi Jal Board", "BWSSB",
    ],

    # ---- ENTERTAINMENT ----
    "entertainment": [
        "Netflix", "Netflix India", "Amazon Prime", "Amazon Prime Video",
        "Disney Plus Hotstar", "Hotstar", "JioCinema",
        "SonyLIV", "Zee5", "ZEE5", "Voot", "ALTBalaji", "MX Player",
        "Spotify", "Spotify India", "Gaana", "JioSaavn", "Wynk Music",
        "YouTube Premium", "Apple Music",
        "BookMyShow", "PVR", "PVR INOX", "INOX", "Cinepolis",
        "Dream11", "MPL", "WinZO", "Ludo King",
    ],

    # ---- BANKS / FINANCIAL INSTITUTIONS ----
    "banking": [
        "SBI", "State Bank of India", "HDFC Bank", "ICICI Bank",
        "Axis Bank", "Kotak Mahindra Bank", "PNB", "Punjab National Bank",
        "Bank of Baroda", "BOB", "Canara Bank", "Union Bank of India",
        "Indian Bank", "Bank of India", "Central Bank of India",
        "IndusInd Bank", "Yes Bank", "RBL Bank", "Federal Bank",
        "South Indian Bank", "Karur Vysya Bank", "City Union Bank",
        "IDFC First Bank", "Bandhan Bank", "AU Small Finance Bank",
        "Ujjivan Small Finance Bank", "Equitas Small Finance Bank",
        "HDFC Securities", "ICICI Securities", "Motilal Oswal",
        "Sharekhan", "Edelweiss", "Kotak Securities",
    ],

    # ---- FMCG / CONSUMER BRANDS ----
    "fmcg": [
        "Amul", "Amul Parlour", "Mother Dairy", "Mother Dairy Booth",
        "ITC", "ITC Store", "HUL", "Hindustan Unilever",
        "Patanjali", "Patanjali Store", "Dabur", "Dabur India",
        "Britannia", "Parle Products", "Parle", "Tata Consumer",
        "Nestle India", "Nestle", "Cadbury", "Mondelez India",
        "Godrej Consumer", "Godrej", "Marico", "Emami",
        "Colgate India", "P&G India", "Reckitt India",
    ],

    # ---- FUEL / AUTOMOTIVE ----
    "fuel_auto": [
        "Indian Oil", "IOCL", "Hindustan Petroleum", "HPCL",
        "Bharat Petroleum", "BPCL", "Reliance Petroleum",
        "Maruti Suzuki", "Hyundai India", "Tata Motors",
        "Mahindra", "Mahindra & Mahindra", "Kia India", "Toyota India",
        "Honda India", "MG Motor", "Skoda India", "Volkswagen India",
        "Hero MotoCorp", "Hero", "TVS Motor", "TVS", "Bajaj Auto",
        "Royal Enfield", "Honda Two Wheelers", "Yamaha India", "Suzuki",
        "CEAT Tyres", "MRF", "Apollo Tyres", "JK Tyre", "Bridgestone",
        "Castrol India", "Shell India", "Gulf Oil",
        "Ather Energy", "Ola Electric", "TVS iQube",
    ],

    # ---- REAL ESTATE / HOME SERVICES ----
    "real_estate": [
        "NoBroker", "MagicBricks", "99acres", "Housing.com",
        "Urban Company", "UrbanClap", "Housejoy",
        "Godrej Properties", "DLF", "Sobha Limited", "Prestige Group",
        "Lodha Group", "Oberoi Realty", "Brigade Group",
        "Livspace", "HomeLane", "DesignCafe", "Homelane",
        "Rentomojo", "Furlenco", "CasaOne",
    ],

    # ---- GOVERNMENT / PUBLIC ----
    "government": [
        "IRCTC", "IRCTC Web", "Indian Railways",
        "Passport Seva", "Passport Office",
        "MCD Tax", "Municipal Tax", "Property Tax",
        "Traffic Police", "Traffic Challan",
        "Income Tax", "Income Tax Department", "GST Portal",
        "Aadhaar", "UIDAI", "Election Commission",
        "NHAI", "FASTag", "NHAI Toll",
        "e-Challan", "Digilocker",
        "State Transport", "KSRTC", "MSRTC", "UPSRTC", "GSRTC", "APSRTC",
        "Delhi Metro", "DMRC", "Mumbai Metro", "Bangalore Metro", "BMRCL",
        "Hyderabad Metro", "Chennai Metro", "Kolkata Metro",
    ],
}


# ============================================================================
# LAYER 2: LOCAL BUSINESS NAME PATTERNS (Combinatorial)
# ============================================================================

# Deity / Auspicious prefixes
DEITY_NAMES = [
    "Shri", "Sri", "Om", "Sai", "Shiv", "Ganesh", "Lakshmi", "Durga",
    "Hanuman", "Krishna", "Balaji", "Vaishnavi", "Mahalakshmi", "Gauri",
    "Radha", "Ram", "Bajrang", "Santoshi", "Jai Mata Di", "Jai Ambey",
    "Sai Ram", "Jai Shri Ram", "Jai Ganesh", "Namo", "Shri Ram",
    "Shree", "Jay", "Jai", "New", "Super", "Royal", "National",
    "City", "Metro", "Modern", "Golden", "Diamond", "Star", "Lucky",
    "Popular", "Standard", "Premier", "Classic", "Excel",
    "Bharat", "Hindustan", "Desh", "Azad", "Tiranga",
    "Al Huda", "Bismillah", "Madina", "Mecca", "Noor",
    "Al Ameen", "Al Fatah", "Al Noor", "Firdaus",
    "Guru Nanak", "Gurdwara", "Khalsa",
    "Mahadev", "Shankar", "Vishnu", "Brahma", "Saraswati", "Parvati",
    "Narayan", "Gopal", "Govind", "Murli", "Radhey", "Sita Ram",
    "Shri Hari", "Shri Krishna", "Shri Ganesh", "Shri Balaji",
    "Jai Durga", "Jai Hanuman", "Jai Shiv", "Jai Lakshmi",
    "Maa Durga", "Maa Lakshmi", "Maa Kali", "Maa Saraswati",
    "Shri Sai", "Om Sai", "Sai Baba", "Shirdi Sai",
    "Tirupati", "Venkateswara", "Ranganath", "Padmavati",
    "Jagannath", "Vitthal", "Pandurang", "Datta", "Dattatreya",
    "Ambika", "Bhavani", "Chamunda", "Vaishno", "Jyoti",
    "Navgraha", "Surya", "Chandra", "Mangal", "Budh",
    "Supreme", "Grand", "Imperial", "Elite", "Prime",
    "Global", "Universal", "Central", "United", "Allied",
    "Pioneer", "Progress", "Success", "Fortune", "Aashirwad",
    "Mangalam", "Subham", "Pavitra", "Divya", "Anand",
    "Sunrise", "Sunshine", "Moonlight", "Rainbow", "Silver",
    "Platinum", "Crystal", "Pearl", "Ruby", "Emerald",
    "Sapphire", "Coral", "Amber", "Topaz", "Opal",
    "Green", "Blue", "Red", "White", "Pink",
    "Angel", "Grace", "Hope", "Faith", "Trust",
    "Al Madina", "Al Noor", "Bismillah", "Rehman", "Raheem",
    "Al Kabir", "Faiz", "Barkat", "Tayyab", "Salam",
    "Guru", "Sant", "Bhai", "Pandit", "Swami",
    "Devi", "Mata", "Ammaji", "Amma", "Thakur",
    "Adarsh", "Ideal", "Model", "Perfect", "Best",
    "Fast", "Quick", "Speed", "Express", "Rapid",
    "Big", "Mega", "Mini", "Micro", "Smart",
    "Digital", "Online", "E-", "i-", "Tech",
    "Eco", "Bio", "Organic", "Natural", "Fresh",
    "Choice", "Select", "Budget", "Value", "Economy",
    "Heritage", "Legacy", "Tradition", "Ancient", "Old",
    "Apna", "Hamara", "Sabka", "Sasta", "Accha",
]

# Person surnames used as business prefixes (creates ambiguity — important!)
BUSINESS_PERSON_NAMES = [
    "Sharma", "Gupta", "Agarwal", "Jain", "Patel", "Shah",
    "Singh", "Kumar", "Verma", "Mishra", "Pandey", "Tiwari",
    "Reddy", "Rao", "Naidu", "Iyer", "Nair", "Pillai",
    "Khan", "Ahmed", "Siddiqui", "Sheikh", "Malik",
    "Banerjee", "Ghosh", "Das", "Mukherjee", "Bose",
    "Patil", "Deshmukh", "Kulkarni", "Jadhav",
    "Mehta", "Desai", "Modi", "Gandhi",
    "Gill", "Sidhu", "Dhillon", "Bedi",
    "Raju", "Babu", "Bhai", "Baba", "Chacha",
    "Ramesh", "Suresh", "Rajesh", "Dinesh", "Mukesh",
    "Amit", "Sanjay", "Vijay", "Ajay", "Deepak",
    "Priya", "Neha", "Sunita", "Asha", "Meena",
    "Saxena", "Srivastava", "Rastogi", "Kapoor", "Malhotra",
    "Arora", "Chopra", "Bhatia", "Khurana", "Tandon",
    "Yadav", "Chauhan", "Rajput", "Thakur", "Rawat",
    "Bhatt", "Joshi", "Negi", "Pant", "Bisht",
    "Dubey", "Shukla", "Dwivedi", "Tripathi", "Chaturvedi",
    "Gowda", "Hegde", "Shetty", "Kamath", "Pai",
    "Chettiar", "Gounder", "Subramanian", "Natarajan",
    "Mohanty", "Sahoo", "Nayak", "Behera", "Dash",
    "Pawar", "Shinde", "More", "Chavan", "Bhosale",
    "Solanki", "Rathod", "Vaghela", "Parmar", "Jadeja",
    "Grewal", "Sandhu", "Bajwa", "Cheema", "Randhawa",
    "Sarkar", "Biswas", "Mondal", "Chakraborty", "Dutta",
    "Soni", "Kothari", "Doshi", "Parekh", "Shroff",
    "Ansari", "Qureshi", "Hashmi", "Mirza", "Pathan",
    "Gaikwad", "Kamble", "Waghmare", "Thorat", "Kale",
    "Manoj", "Anil", "Sunil", "Ashok", "Rakesh",
    "Rohit", "Mohit", "Vikas", "Rahul", "Nikhil",
    "Ankit", "Gaurav", "Sachin", "Tushar", "Yogesh",
    "Pooja", "Anjali", "Kavita", "Rekha", "Divya",
    "Lakshmi", "Sarita", "Geeta", "Kamla", "Pushpa",
    "Pappu", "Guddu", "Munna", "Chhotu", "Sonu",
    "Rinku", "Pintu", "Chintu", "Bablu", "Goldy",
    "Kaka", "Dada", "Nana", "Mama", "Tau",
    "Lala", "Lalji", "Sethji", "Sahab", "Haji",
]

# Business type suffixes
BUSINESS_SUFFIXES = {
    "local_shop": [
        "General Store", "Kirana Store", "Kirana", "Provision Store",
        "Grocery", "Grocery Store", "Mart", "Super Mart", "Supermarket",
        "Mini Mart", "Daily Needs", "Departmental Store",
        "Store", "Stores", "Shop", "Emporium", "Depot", "Centre",
    ],
    "restaurant": [
        "Restaurant", "Hotel", "Dhaba", "Bhojanalaya", "Cafe",
        "Canteen", "Mess", "Kitchen", "Food Corner", "Food Point",
        "Vaishno Dhaba", "Family Restaurant", "Non Veg Corner",
        "Biryani House", "Biryani Corner", "Chicken Corner",
        "Sweets", "Sweet House", "Mithai", "Mithai Bhandar",
        "Bakery", "Cake Shop", "Confectionery",
        "Juice Corner", "Juice Centre", "Tea Stall", "Chai Point",
        "Snack Bar", "Fast Food", "Chaat Corner", "Paan Shop",
    ],
    "electronics": [
        "Electronics", "Mobiles", "Mobile Point", "Mobile Shop",
        "Communication", "Telecom", "Computers", "Computer Centre",
        "Tech", "IT Solutions", "Digital", "Gadgets",
        "TV Repair", "AC Service", "Refrigeration",
    ],
    "clothing": [
        "Textiles", "Cloth House", "Cloth Store", "Readymade",
        "Readymade Garments", "Fashion", "Fashion Hub", "Boutique",
        "Saree Centre", "Saree House", "Saree Emporium",
        "Fabrics", "Dress Material", "Tailors", "Tailor",
        "Silks", "Silk House", "Vastra Bhandar",
    ],
    "medical": [
        "Medical Store", "Medicals", "Pharmacy", "Drug House",
        "Chemist", "Hospital", "Clinic", "Nursing Home",
        "Dental Clinic", "Eye Clinic", "Ortho Clinic",
        "Pathology Lab", "Diagnostics", "Health Centre",
        "Medical Agency", "Surgical", "Ayurvedic",
    ],
    "hardware": [
        "Hardware", "Hardware Store", "Paint House", "Paints",
        "Sanitary", "Sanitary Ware", "Plumbing", "Electricals",
        "Building Material", "Cement Store", "Steel",
        "Glass House", "Timber", "Ply & Hardware",
    ],
    "auto": [
        "Auto Parts", "Auto", "Motors", "Motor Parts",
        "Garage", "Workshop", "Tyre House", "Tyres",
        "Car Wash", "Service Centre", "Spare Parts",
        "Two Wheeler", "Bike Point", "Cycle Store",
    ],
    "salon": [
        "Beauty Parlour", "Beauty Salon", "Salon",
        "Hair Cutting", "Hair Studio", "Barber Shop",
        "Unisex Salon", "Spa", "Herbal Beauty",
        "Bridal Studio", "Makeover Studio",
    ],
    "jewellery": [
        "Jewellers", "Jewellery", "Gold House", "Bullion",
        "Silver House", "Gems", "Ornaments",
        "Gold & Silver", "Precious", "Diamond",
    ],
    "education": [
        "Coaching", "Coaching Classes", "Coaching Centre",
        "Academy", "Institute", "Tuition Centre", "Classes",
        "Educational", "Learning Centre", "School",
        "Vidyalaya", "Vidya Mandir", "Public School",
    ],
    "professional": [
        "& Associates", "& Co", "Consultants", "Advisory",
        "Legal Services", "Law Firm", "Tax Consultant",
        "CA Firm", "Chartered Accountants", "Architects",
        "Interior", "Design Studio", "Solutions",
    ],
    "travel": [
        "Travels", "Tour & Travels", "Tourism",
        "Transport", "Logistics", "Cargo", "Movers & Packers",
        "Courier", "Courier Service", "Express",
    ],
    "furniture": [
        "Furniture", "Furniture House", "Sofa House",
        "Mattress", "Home Decor", "Furnishing",
        "Curtains", "Interiors", "Kitchen",
    ],
    "stationery": [
        "Stationery", "Stationers", "Book Store", "Book House",
        "Xerox", "Xerox & Printing", "Printing Press",
        "Photo Studio", "Photo Lab", "Digital Studio",
    ],
    "fuel": [
        "Petrol Pump", "Filling Station", "Fuel Station",
        "Gas Agency", "LPG Gas", "CNG Station",
        "Petroleum", "Service Station",
    ],
    "other": [
        "Traders", "Trading Co", "Enterprises", "Enterprise",
        "Pvt Ltd", "Private Limited", "LLP",
        "& Sons", "Bros", "Brothers", "& Company", "Co",
        "Industries", "Manufacturing", "Works",
        "Services", "Solutions", "Agency", "Associates",
        "Gym", "Fitness", "Sports", "Yoga Centre",
        "Opticals", "Optics", "Eye Wear",
        "Laundry", "Dry Cleaners", "Ironing",
        "Nursery", "Flower Shop", "Gift Shop", "Gift House",
        "Pet Shop", "Aquarium", "Kennel",
    ],
}

# "Ki Dukan" / "Wala" patterns
DESI_PATTERNS = {
    "ki_dukan": [
        "{name} Ki Dukan", "{name} Ka Store", "{name} Ka Shop",
    ],
    "wala": [
        "{product} Wala", "{product} Wale", "{product} Corner",
        "{product} Point", "{product} Hub", "{product} Zone",
        "{product} World", "{product} King", "{product} Palace",
    ],
}

PRODUCTS_FOR_WALA = [
    "Chai", "Samosa", "Paan", "Doodh", "Lassi", "Kulfi",
    "Momos", "Juice", "Fruit", "Sabzi", "Flower", "Bartan",
    "Kapda", "Juta", "Cycle", "Mobile", "Laptop", "Computer",
    "Watch", "Glasses", "Bag", "Shoe", "Phool", "Mithai",
    "Chicken", "Fish", "Egg", "Paneer", "Chaat", "Golgappa",
    "Ice Cream", "Cold Drink", "Pizza", "Burger", "Noodle",
    "Biryani", "Tandoori", "Kebab", "Roll", "Tikka",
]


# ============================================================================
# LAYER 3: LEGAL ENTITY NAMES
# ============================================================================

LEGAL_ENTITIES = [
    "Bundl Technologies Pvt Ltd",  # Swiggy
    "Zomato Ltd", "Zomato Private Limited",
    "Flipkart Internet Pvt Ltd", "Flipkart India Pvt Ltd",
    "Amazon Seller Services Pvt Ltd", "Amazon Development Centre India",
    "One97 Communications Ltd",  # Paytm
    "ANI Technologies Pvt Ltd",  # Ola
    "Uber India Systems Pvt Ltd",
    "Le Travenues Technology Pvt Ltd",  # Ixigo
    "PhonePe Pvt Ltd", "PhonePe Private Limited",
    "Oravel Stays Ltd",  # OYO
    "Think & Learn Pvt Ltd",  # BYJU'S
    "MakeMyTrip India Pvt Ltd",
    "Jasper Infotech Pvt Ltd",  # Snapdeal
    "Fashnear Technologies Pvt Ltd",  # Meesho
    "Kiranakart Technologies Pvt Ltd",  # Zepto
    "Grofers India Pvt Ltd",  # Blinkit (old name)
    "Locobuzz Solutions Pvt Ltd",
    "Razorpay Software Pvt Ltd",
    "Resilient Innovations Pvt Ltd",  # BharatPe
    "Curefit Healthcare Pvt Ltd",
    "Practo Technologies Pvt Ltd",
    "Info Edge India Ltd",  # Naukri/99acres
    "FSN E-Commerce Ventures Ltd",  # Nykaa
    "Lenskart Solutions Pvt Ltd",
    "Avenue Supermarts Ltd",  # DMart
    "Reliance Retail Ventures Ltd",
    "Trent Ltd",  # Westside
    "Aditya Birla Fashion and Retail Ltd",
    "Titan Company Ltd",
    "Jubilant Foodworks Ltd",  # Dominos
    "Westlife Foodworld Ltd",  # McDonalds India
    "Devyani International Ltd",  # KFC/Pizza Hut
    "Restaurant Brands Asia Ltd",  # Burger King
    "Sapphire Foods India Ltd",  # KFC/Pizza Hut/Taco Bell
    "Tata Consumer Products Ltd",
    "Hindustan Unilever Ltd",
    "ITC Limited",
    "Nestle India Ltd",
    "Marico Ltd",
    "Dabur India Ltd",
    "Godrej Consumer Products Ltd",
    "Britannia Industries Ltd",
    "Adani Green Energy Ltd",
    "Bharti Airtel Ltd",
    "Reliance Industries Ltd",
    "Tata Consultancy Services Ltd", "TCS",
    "Infosys Ltd", "Infosys",
    "Wipro Ltd", "Wipro",
    "HCL Technologies Ltd",
    "Tech Mahindra Ltd",
]


# ============================================================================
# LAYER 5: SYNTHETIC / MODERN BRAND-STYLE NAMES
# ============================================================================

SYNTHETIC_BRANDS = [
    # Fintech-style
    "PaySmart", "QuickPay", "EasyPay", "FastCash", "CashNow",
    "RuPay Merchant", "DigiPay", "SmartPay", "InstaPay", "NowPay",
    "PayEase", "MoneyWise", "WealthFirst", "LoanBazaar", "CreditKaro",
    "LendingTree India", "FinBox", "FlexiLoans", "Capital Float",

    # Indian-flavored modern
    "DesiMart", "BharatPay", "SwadeshiStore", "ApnaMart", "ApnaDukan",
    "JanataStore", "GharKart", "SabkiDukan", "DailyBasket", "KisanMart",
    "SabjiBazaar", "MediBuddy", "HealthyKart", "FitIndia", "YogaKart",
    "EduKart", "SkillShala", "PathShala", "VidyaOnline", "GuruKool",

    # Tech/startup style
    "ShopKaro", "BuyZone", "CartFresh", "GoKart", "DashMart",
    "ClickOrder", "TapBuy", "SwipeShop", "ScanPay", "QRMerchant",
    "OneStop", "AllInOne", "MultiMart", "MegaStore", "UltraMart",
    "QuickBite", "FoodRush", "MealBox", "TiffinWala", "HomePlate",
    "RideNow", "GoRide", "AutoBook", "CabNow", "DriveEasy",
]


# ============================================================================
# NAME GENERATION LOGIC
# ============================================================================

def generate_real_brand_entries() -> list[dict]:
    """Generate entries from real brand names with format variants."""
    entries = []
    for category, brands in REAL_BRANDS.items():
        for brand in brands:
            # Original
            entries.append({
                "name": brand,
                "category": category,
                "source": "real_brand",
            })
            # ALL CAPS
            entries.append({
                "name": brand.upper(),
                "category": category,
                "source": "real_brand",
            })
            # lowercase
            entries.append({
                "name": brand.lower(),
                "category": category,
                "source": "real_brand",
            })
    return entries


def generate_local_business_names(count_per_type: int = 300, seed: int = 42) -> list[dict]:
    """Generate local business names using combinatorial patterns."""
    random.seed(seed)
    entries = []

    for biz_type, suffixes in BUSINESS_SUFFIXES.items():
        for _ in range(count_per_type):
            # Pick prefix: deity name OR person name
            if random.random() < 0.5:
                prefix = random.choice(DEITY_NAMES)
            else:
                prefix = random.choice(BUSINESS_PERSON_NAMES)

            suffix = random.choice(suffixes)
            name = f"{prefix} {suffix}"

            entries.append({
                "name": name,
                "category": biz_type if biz_type != "other" else "local_shop",
                "source": "local_pattern",
            })

            # Variant: ALL CAPS
            if random.random() < 0.3:
                entries.append({
                    "name": name.upper(),
                    "category": biz_type if biz_type != "other" else "local_shop",
                    "source": "local_pattern",
                })

    return entries


def generate_desi_patterns(count: int = 500, seed: int = 43) -> list[dict]:
    """Generate 'Ki Dukan' / 'Wala' style names."""
    random.seed(seed)
    entries = []

    # Ki Dukan patterns
    for _ in range(count // 2):
        name = random.choice(BUSINESS_PERSON_NAMES)
        pattern = random.choice(DESI_PATTERNS["ki_dukan"])
        biz_name = pattern.format(name=name)
        entries.append({
            "name": biz_name,
            "category": "local_shop",
            "source": "desi_pattern",
        })

    # Wala patterns
    for _ in range(count // 2):
        product = random.choice(PRODUCTS_FOR_WALA)
        pattern = random.choice(DESI_PATTERNS["wala"])
        biz_name = pattern.format(product=product)
        entries.append({
            "name": biz_name,
            "category": "local_shop",
            "source": "desi_pattern",
        })

        # Also with person name: "Raju Chai Wala"
        if random.random() < 0.4:
            person = random.choice(BUSINESS_PERSON_NAMES)
            entries.append({
                "name": f"{person} {biz_name}",
                "category": "local_shop",
                "source": "desi_pattern",
            })

    return entries


def generate_legal_entity_entries() -> list[dict]:
    """Generate entries from legal entity names."""
    entries = []
    for name in LEGAL_ENTITIES:
        entries.append({
            "name": name,
            "category": "corporate",
            "source": "legal_entity",
        })
        entries.append({
            "name": name.upper(),
            "category": "corporate",
            "source": "legal_entity",
        })
    return entries


def generate_synthetic_entries() -> list[dict]:
    """Generate entries from synthetic brand names."""
    entries = []
    for name in SYNTHETIC_BRANDS:
        entries.append({
            "name": name,
            "category": "other_merchant",
            "source": "synthetic",
        })
        entries.append({
            "name": name.upper(),
            "category": "other_merchant",
            "source": "synthetic",
        })
    return entries


def load_llm_generated_names(path: str) -> list[dict]:
    """Load LLM-generated merchant names from JSON."""
    if not os.path.exists(path):
        print(f"  [SKIP] LLM file not found: {path}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = []
    for category, names in data.items():
        for name in names:
            entries.append({
                "name": name,
                "category": category,
                "source": "llm_generated",
            })
    return entries


# ============================================================================
# AUGMENTATION (same corruption pipeline as person names)
# ============================================================================

def truncate_name(name, max_len=None):
    max_len = max_len or random.randint(12, 25)
    if len(name) <= max_len:
        return name
    return name[:max_len].rstrip()

def truncate_with_ellipsis(name):
    max_len = random.randint(10, 20)
    if len(name) <= max_len:
        return name
    suffix = random.choice(["...", "..", "."])
    return name[:max_len].rstrip() + suffix

def drop_random_chars(name):
    if len(name) < 4: return name
    chars = list(name)
    droppable = [i for i, c in enumerate(chars) if c not in ' ']
    if len(droppable) < 2: return name
    indices = random.sample(droppable, min(2, len(droppable)))
    return ''.join(c for i, c in enumerate(chars) if i not in indices)

def random_case_noise(name):
    return ''.join(c.upper() if random.random() < 0.4 else c.lower() for c in name)

def add_extra_spaces(name):
    parts = name.split()
    if len(parts) < 2: return name
    return ('  '.join(parts) if random.random() < 0.5
            else '   '.join(parts))

def remove_spaces(name):
    return name.replace(' ', '')

def replace_space_with_char(name):
    return name.replace(' ', random.choice(['_', '.', '-', '/']))

def common_typos(name):
    if len(name) < 4: return name
    chars = list(name)
    positions = [i for i in range(len(chars) - 1) if chars[i] != ' ' and chars[i+1] != ' ']
    if not positions: return name
    pos = random.choice(positions)
    chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
    return ''.join(chars)

def add_prefix_noise(name):
    prefix = random.choice(["UPI-", "UPI/", "NEFT-", "IMPS-", "P2P-", "PAY-"])
    return prefix + name

def add_suffix_noise(name):
    suffix = random.choice([
        str(random.randint(1, 999)),
        "-" + str(random.randint(100, 999)),
        "/" + str(random.randint(10, 99)),
        " " + ''.join(random.choices(string.ascii_uppercase, k=3)),
        "-" + ''.join(random.choices(string.digits, k=4)),
    ])
    return name + suffix

def insert_random_char(name):
    if len(name) < 3: return name
    pos = random.randint(1, len(name) - 1)
    return name[:pos] + random.choice(string.ascii_lowercase) + name[pos:]

def repeat_a_char(name):
    if len(name) < 3: return name
    positions = [i for i, c in enumerate(name) if c.isalpha()]
    if not positions: return name
    pos = random.choice(positions)
    return name[:pos] + name[pos] + name[pos:]

def abbreviate_words(name):
    parts = name.split()
    n = random.randint(2, 4)
    return ' '.join(p[:n] for p in parts)

def partial_mask(name):
    if len(name) < 5: return name
    visible = random.randint(3, min(8, len(name) - 2))
    return name[:visible] + random.choice(['***', '...', '..'])

MERCHANT_AUGMENTATIONS = [
    (truncate_name, 3), (truncate_with_ellipsis, 2),
    (drop_random_chars, 3), (random_case_noise, 2),
    (add_extra_spaces, 2), (remove_spaces, 2),
    (replace_space_with_char, 2), (common_typos, 3),
    (add_prefix_noise, 2), (add_suffix_noise, 2),
    (insert_random_char, 1), (repeat_a_char, 1),
    (abbreviate_words, 2), (partial_mask, 1),
]


def augment_merchant(name: str, num_augments: int = 3) -> list[str]:
    """Apply random augmentations to a merchant name."""
    pool = []
    for func, weight in MERCHANT_AUGMENTATIONS:
        pool.extend([func] * weight)

    results = []
    for _ in range(num_augments):
        func = random.choice(pool)
        aug = func(name)
        if aug and aug.strip() and aug.lower().strip() != name.lower().strip():
            results.append(aug.strip())

    # 20% chance of compound corruption
    if random.random() < 0.2:
        f1 = random.choice(pool)
        f2 = random.choice(pool)
        compound = f2(f1(name))
        if compound and compound.strip():
            results.append(compound.strip())

    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_full_dataset(seed: int = 42) -> list[dict]:
    """Generate the complete merchant names dataset."""
    random.seed(seed)
    all_entries = []
    seen = set()

    def add_entries(entries, label="MERCHANT"):
        for e in entries:
            key = e["name"].lower().strip()
            if key and key not in seen and len(key) > 1:
                seen.add(key)
                all_entries.append({
                    "name": e["name"].strip(),
                    "label": label,
                    "category": e.get("category", "other_merchant"),
                    "source": e.get("source", "unknown"),
                    "augmentation": "original",
                })

    # Layer 1: Real brands
    print("Layer 1: Real brands...")
    brand_entries = generate_real_brand_entries()
    add_entries(brand_entries)
    print(f"  -> {len([e for e in all_entries if e['source'] == 'real_brand'])} entries")

    # Layer 2: Local business patterns
    print("Layer 2: Local business names...")
    local_entries = generate_local_business_names(count_per_type=2000)
    add_entries(local_entries)
    print(f"  -> {len([e for e in all_entries if e['source'] == 'local_pattern'])} entries")

    # Layer 2b: Desi patterns (Ki Dukan, Wala)
    print("Layer 2b: Desi patterns...")
    desi_entries = generate_desi_patterns(count=5000)
    add_entries(desi_entries)
    print(f"  -> {len([e for e in all_entries if e['source'] == 'desi_pattern'])} entries")

    # Layer 3: Legal entities
    print("Layer 3: Legal entities...")
    legal_entries = generate_legal_entity_entries()
    add_entries(legal_entries)
    print(f"  -> {len([e for e in all_entries if e['source'] == 'legal_entity'])} entries")

    # Layer 5: Synthetic brands
    print("Layer 5: Synthetic brands...")
    synth_entries = generate_synthetic_entries()
    add_entries(synth_entries)
    print(f"  -> {len([e for e in all_entries if e['source'] == 'synthetic'])} entries")

    # Layer 6: LLM-generated
    print("Layer 6: LLM-generated names...")
    llm_path = os.path.join(os.path.dirname(__file__), "llm_generated_merchants.json")
    llm_entries = load_llm_generated_names(llm_path)
    add_entries(llm_entries)
    print(f"  -> {len([e for e in all_entries if e['source'] == 'llm_generated'])} entries")

    clean_count = len(all_entries)
    print(f"\nTotal clean entries: {clean_count}")

    # Layer 7: Augmentation
    print("Layer 7: Augmenting...")
    augmented_entries = []
    for entry in list(all_entries):  # iterate over copy
        aug_names = augment_merchant(entry["name"], num_augments=4)
        for aug_name in aug_names:
            aug_key = aug_name.lower().strip()
            if aug_key not in seen and aug_key:
                seen.add(aug_key)
                augmented_entries.append({
                    "name": aug_name,
                    "label": "MERCHANT",
                    "category": entry["category"],
                    "source": entry["source"],
                    "augmentation": "augmented",
                })

    all_entries.extend(augmented_entries)
    print(f"  -> {len(augmented_entries)} augmented entries")
    print(f"\nTotal (clean + augmented): {len(all_entries)}")

    # Shuffle
    random.shuffle(all_entries)
    return all_entries


def save_dataset(dataset: list[dict], output_path: str):
    """Save to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["name", "label", "category", "source", "augmentation"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print(f"[OK] Saved {len(dataset)} merchant names to {output_path}")


def print_stats(dataset: list[dict]):
    """Print statistics."""
    from collections import Counter

    print("\n" + "=" * 60)
    print("MERCHANT DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal entries: {len(dataset)}")

    # Original vs augmented
    orig = sum(1 for r in dataset if r['augmentation'] == 'original')
    aug = len(dataset) - orig
    print(f"\n  Original (clean):  {orig:6d}")
    print(f"  Augmented (noisy): {aug:6d}")

    # By source
    src_counts = Counter(r["source"] for r in dataset)
    print(f"\nBy Source:")
    for src, count in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:25s} -> {count:5d}")

    # By category
    cat_counts = Counter(r["category"] for r in dataset)
    print(f"\nBy Category ({len(cat_counts)}):")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s} -> {count:5d}")

    # Sample
    print(f"\nRandom sample (30):")
    print("-" * 80)
    sample = random.sample(dataset, min(30, len(dataset)))
    for row in sample:
        tag = "CLEAN" if row['augmentation'] == 'original' else "NOISY"
        print(f"  [{tag}] {row['name']:40s} | {row['category']:15s} | {row['source']}")


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "indian_merchant_names.csv")
    dataset = generate_full_dataset(seed=42)
    save_dataset(dataset, output_path)
    print_stats(dataset)

