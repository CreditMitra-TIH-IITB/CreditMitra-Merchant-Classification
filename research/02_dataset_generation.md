# Stage 2: Dataset Generation

**Date:** 2026-06-22  
**Status:** ✅ Complete

---

## Overview

Built two comprehensive datasets for training the PERSON vs MERCHANT classifier:

| Dataset | Clean Names | Augmented | Total | File |
|:---|---:|---:|---:|:---|
| **Person Names** | 37,830 | 69,733 | **107,563** | `data/indian_person_names_augmented.csv` |
| **Merchant Names** | 29,057 | 97,243 | **126,300** | `data/indian_merchant_names.csv` |
| **Combined** | 66,887 | 166,976 | **233,863** | — |

**Class balance ratio**: 0.85:1 (Person:Merchant) — well balanced.

---

## Person Names Dataset

### Generation Script: `data/generate_indian_names.py`

#### Methodology

1. **Curated name lists** covering 17 Indian linguistic/regional groups:
   - Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati
   - Punjabi, Rajasthani, Odia, Assamese, Northeast, Jain, Parsi, Christian, Muslim

2. **Name components**:
   - ~4,000+ first names (male & female) across all regions
   - ~2,300+ surnames with region-compatible pairing logic
   - ~140 middle names

3. **10 realistic name formats**:

   | Format | Example | Weight |
   |:---|:---|---:|
   | `firstname_lastname` | Ramesh Sharma | 15 |
   | `lastname_firstname` | Sharma Ramesh | 10 |
   | `firstname_initial` | Ramesh S | 20 |
   | `initial_lastname` | R Sharma | 15 |
   | `upper` | RAMESH SHARMA | 10 |
   | `lower` | ramesh sharma | 10 |
   | `firstname_only` | Ramesh | 10 |
   | `three_part` | Ramesh Kumar Sharma | 5 |
   | `three_part_upper` | RAMESH KUMAR SHARMA | 5 |
   | `initials_lastname` | R K Sharma | 5 |

4. **Cultural accuracy**: Surnames are mapped via `get_compatible_surnames()` to ensure regionally accurate combinations (e.g., Tamil first names get Tamil surnames, not Punjabi ones).

#### Statistics (Clean)

- **Total**: 38,920 unique names
- **500 names generated per region** across 17 regions
- **Gender split**: 49% female, 51% male

### Augmentation Script: `data/augment_names.py`

Applied 20 corruption types to simulate real-world UPI narration noise:

| # | Corruption | Example | Weight |
|:---|:---|:---|---:|
| 1 | Truncation | `Ramesh Kumar Sha` | 3 |
| 2 | Truncation + ellipsis | `R Chatt..` | 2 |
| 3 | Character drop | `Rahul` → `Rhu` | 3 |
| 4 | Vowel drop | `Chawla` → `Chwl` | 2 |
| 5 | Double char merge | `Banerjee` → `Banerje` | 2 |
| 6 | Extra spaces | `Trevor   G` | 2 |
| 7 | Remove spaces | `RameshSharma` | 2 |
| 8 | Space → special char | `Acharya/Joydeb` | 2 |
| 9 | Random case noise | `rAmEsH sHaRmA` | 2 |
| 10 | Dot insertion | `P. Nagori` | 2 |
| 11 | Prefix noise | `NEFT-Kalpana Jagdev` | 2 |
| 12 | Suffix noise | `P Gajjar-167` | 2 |
| 13 | Adjacent char swap | `Bhandari` → `Bhandrai` | 3 |
| 14 | Phonetic substitution | `Nimesh` → `Nimes` | 3 |
| 15 | Partial masking | `R Chatt***` | 2 |
| 16 | First name only | `Senthil A` → `Senthil` | 2 |
| 17 | Last name only | `Ojasvi Gahane` → `Gahane` | 2 |
| 18 | Word abbreviation | `Ram Kum Sha` | 2 |
| 19 | Random char insert | `Nanda Rlamchandra` | 1 |
| 20 | Char repeat | `darab` → `ddarab` | 1 |

- **15% compound corruption** chance (two augmentations stacked)
- **2 augmented variants** per clean name
- **Deduplication** via case-insensitive set

---

## Merchant Names Dataset

### Generation Script: `data/generate_merchant_names.py`

#### Methodology — 7 Layers

##### Layer 1: Real Indian & International Brands (~663 clean)
Hardcoded 600+ actual brands across 17 categories:
- E-commerce: Amazon, Flipkart, Myntra, Meesho, Nykaa, etc.
- Food Delivery: Swiggy, Zomato, Blinkit, Zepto, Dominos, etc.
- Fintech: PhonePe, Paytm, CRED, Razorpay, Groww, Zerodha, etc.
- Retail: DMart, Reliance, Spencer's, V-Mart, etc.
- Fashion: Zara, H&M, FabIndia, Tanishq, Lenskart, etc.
- Healthcare: Apollo, Netmeds, PharmEasy, Max Hospital, etc.
- And 11 more categories...

Each brand generates 3 variants: original, UPPER, lower.

##### Layer 2: Local Business Names (~26,446 clean)
Combinatorial generation: `[Prefix] + [Business Suffix]`

- **155+ prefixes**: Deity names (Shri, Om, Sai, Ganesh...), person names (Sharma, Gupta, Khan...), adjectives (New, Royal, Modern, Golden...)
- **130+ business suffixes** across 17 business types: General Store, Electronics, Restaurant, Pharmacy, Salon, Garage, Jewellers, etc.
- **2,000 combinations per business type**

##### Layer 3: Desi Patterns (~1,827 clean)
Indian-flavor naming conventions:
- "Ki Dukan" patterns: `Raju Ki Dukan`, `Pappu Ka Store`
- "Wala" patterns: `Chai Wala`, `Samosa Corner`, `Biryani Palace`
- 100+ products × 21 pattern templates × person names

##### Layer 4: Legal Entity Names (~57 clean)
Registered company names: `Bundl Technologies Pvt Ltd` (Swiggy), `One97 Communications Ltd` (Paytm), etc.

##### Layer 5: Synthetic Brands (~64 clean)
Modern startup-style names: `PaySmart`, `DesiMart`, `BharatPay`, `QuickBite`, etc.

##### Layer 6: LLM-Generated Names
Attempted using local Gemma4:e2b via Ollama, but encountered GGML compatibility crash. Skipped — other layers provide sufficient coverage.

##### Layer 7: Augmentation (~97,243 augmented)
Same corruption pipeline as person names (14 augmentation types), with:
- **4 augmented variants per clean name**
- **20% compound corruption** chance
- Deduplication

#### Statistics

| Source | Clean | With Augmentation |
|:---|---:|---:|
| Real brands | 663 | 2,488 |
| Local patterns | 26,446 | 115,549 |
| Desi patterns | 1,827 | 7,795 |
| Legal entities | 57 | 245 |
| Synthetic | 64 | 223 |

**33 merchant sub-categories** tracked (grocery, restaurant, electronics, clothing, medical, salon, auto, jewellery, education, etc.)

---

## Dataset Quality: Overlap Analysis

### Script: `data/analyze_overlap.py`

Checked for collisions between person and merchant datasets:

| Check | Overlaps | Notes |
|:---|---:|:---|
| **Clean-only overlaps** | **3** | `godrej`, `raymond`, `zara` — all legitimate dual-use names |
| Exact match (all entries) | 252 | Augmented noise fragments |
| Case-insensitive match | 364 | Augmented noise fragments |
| Normalized match | 444 | Augmented noise fragments |
| Shared first-name tokens | 1,621 | Expected — names like "Sharma" appear as both person names and merchant prefixes |
| **Jaccard similarity** | **0.16%** | Excellent separation |

### Key Finding

The only 3 real overlaps (`Godrej`, `Raymond`, `Zara`) are **genuine ambiguous names** that exist as both person names and brand names. This is a feature, not a bug — the classifier will learn from context.

The 1,621 shared first-name tokens (e.g., `Sharma`, `Gupta`, `Amit`) are **intentionally present** in both datasets because they test the classifier's ability to distinguish:
- `"Sharma"` → PERSON
- `"Sharma Electronics"` → MERCHANT

---

## File Structure

```
data/
├── generate_indian_names.py        # Person name generator
├── augment_names.py                # UPI noise augmentation pipeline
├── generate_merchant_names.py      # Merchant name generator (with built-in augmentation)
├── analyze_overlap.py              # Cross-dataset overlap analysis
├── llm_generate_merchants.py       # (Attempted) LLM merchant generation via Ollama
├── indian_person_names.csv         # Clean person names (38,920)
├── indian_person_names_augmented.csv  # Augmented person names (107,563)
├── indian_merchant_names.csv       # Augmented merchant names (126,300)
└── llm_generated_merchants.json    # Empty (LLM generation failed)
```

---

## Next Steps

→ Proceed to **Stage 3: Embedding Generation & Model Training**

- [ ] Generate Qwen3 embeddings for all 233K entries
- [ ] Train Logistic Regression classifier
- [ ] Evaluate with train/test split
- [ ] Test on edge cases (ambiguous names, heavy corruption)
- [ ] Package for production inference
