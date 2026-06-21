"""
Overlap Analysis: Person Names vs Merchant Names
=================================================
Check for exact matches, case-insensitive matches, and normalized matches
between the two datasets to identify potential classifier confusion.
"""

import csv
import os


def normalize(s):
    """Strip all spaces, lowercase."""
    return ''.join(s.lower().split())


def load_names(path, name_col='name'):
    """Load names from CSV."""
    with open(path, encoding='utf-8') as f:
        return [row[name_col].strip() for row in csv.DictReader(f)]


def main():
    base = os.path.dirname(__file__)
    person_path = os.path.join(base, "indian_person_names_augmented.csv")
    merchant_path = os.path.join(base, "indian_merchant_names.csv")

    print("Loading datasets...")
    persons = load_names(person_path)
    merchants = load_names(merchant_path)
    print(f"  Person names:   {len(persons):,}")
    print(f"  Merchant names: {len(merchants):,}")

    # ============================================================
    # 1. EXACT MATCH (case-sensitive)
    # ============================================================
    person_set = set(persons)
    merchant_set = set(merchants)
    exact_overlap = person_set & merchant_set

    print(f"\n{'='*60}")
    print("1. EXACT MATCH (case-sensitive)")
    print(f"{'='*60}")
    print(f"Overlapping entries: {len(exact_overlap)}")
    if exact_overlap:
        for s in sorted(list(exact_overlap))[:40]:
            print(f"  - '{s}'")
        if len(exact_overlap) > 40:
            print(f"  ... and {len(exact_overlap) - 40} more")

    # ============================================================
    # 2. CASE-INSENSITIVE MATCH
    # ============================================================
    person_lower = {}  # lowercase -> original
    for p in persons:
        key = p.lower().strip()
        if key not in person_lower:
            person_lower[key] = p

    merchant_lower = {}
    for m in merchants:
        key = m.lower().strip()
        if key not in merchant_lower:
            merchant_lower[key] = m

    ci_overlap = set(person_lower.keys()) & set(merchant_lower.keys())

    print(f"\n{'='*60}")
    print("2. CASE-INSENSITIVE MATCH")
    print(f"{'='*60}")
    print(f"Overlapping entries: {len(ci_overlap)}")
    if ci_overlap:
        for s in sorted(list(ci_overlap))[:60]:
            p_orig = person_lower[s]
            m_orig = merchant_lower[s]
            print(f"  PERSON: '{p_orig:30s}'  <->  MERCHANT: '{m_orig}'")
        if len(ci_overlap) > 60:
            print(f"  ... and {len(ci_overlap) - 60} more")

    # ============================================================
    # 3. NORMALIZED MATCH (no spaces, lowercase)
    # ============================================================
    person_norm = {}
    for p in persons:
        key = normalize(p)
        if key not in person_norm:
            person_norm[key] = p

    merchant_norm = {}
    for m in merchants:
        key = normalize(m)
        if key not in merchant_norm:
            merchant_norm[key] = m

    norm_overlap = set(person_norm.keys()) & set(merchant_norm.keys())

    print(f"\n{'='*60}")
    print("3. NORMALIZED MATCH (no spaces, lowercase)")
    print(f"{'='*60}")
    print(f"Overlapping entries: {len(norm_overlap)}")
    if norm_overlap:
        for s in sorted(list(norm_overlap))[:60]:
            p_orig = person_norm[s]
            m_orig = merchant_norm[s]
            print(f"  PERSON: '{p_orig:30s}'  <->  MERCHANT: '{m_orig}'")
        if len(norm_overlap) > 60:
            print(f"  ... and {len(norm_overlap) - 60} more")

    # ============================================================
    # 4. SUBSTRING / TOKEN OVERLAP
    # ============================================================
    # Check if any person name appears AS A SUBSTRING in merchant names
    # (This catches "Ramesh" appearing in "Ramesh Electronics")
    # Only check clean person first-names (short ones)
    print(f"\n{'='*60}")
    print("4. TOKEN-LEVEL ANALYSIS")
    print(f"{'='*60}")

    # Get unique person first names (single-word names)
    person_first_names = set()
    for p in persons:
        parts = p.split()
        if len(parts) == 1 and len(p) > 2:
            person_first_names.add(p.lower())
        elif len(parts) >= 1 and len(parts[0]) > 2:
            person_first_names.add(parts[0].lower())

    # Get unique merchant first tokens
    merchant_first_tokens = set()
    for m in merchants:
        parts = m.split()
        if len(parts) >= 1 and len(parts[0]) > 2:
            merchant_first_tokens.add(parts[0].lower())

    token_overlap = person_first_names & merchant_first_tokens
    print(f"Person first-name tokens: {len(person_first_names):,}")
    print(f"Merchant first tokens:    {len(merchant_first_tokens):,}")
    print(f"Shared tokens:            {len(token_overlap):,}")
    print(f"\nThese are names that appear as BOTH a person AND a merchant prefix:")
    for t in sorted(list(token_overlap))[:80]:
        print(f"  - '{t}'")
    if len(token_overlap) > 80:
        print(f"  ... and {len(token_overlap) - 80} more")

    # ============================================================
    # 5. OVERALL SCORE
    # ============================================================
    unique_persons = len(set(p.lower() for p in persons))
    unique_merchants = len(set(m.lower() for m in merchants))
    union = len(set(p.lower() for p in persons) | set(m.lower() for m in merchants))
    jaccard = len(ci_overlap) / union * 100 if union > 0 else 0

    print(f"\n{'='*60}")
    print("OVERLAP SCORE SUMMARY")
    print(f"{'='*60}")
    print(f"  Unique person names:        {unique_persons:>9,}")
    print(f"  Unique merchant names:      {unique_merchants:>9,}")
    print(f"  Exact overlaps:             {len(exact_overlap):>9,}  ({len(exact_overlap)/min(unique_persons, unique_merchants)*100:.3f}%)")
    print(f"  Case-insensitive overlaps:  {len(ci_overlap):>9,}  ({len(ci_overlap)/min(unique_persons, unique_merchants)*100:.3f}%)")
    print(f"  Normalized overlaps:        {len(norm_overlap):>9,}  ({len(norm_overlap)/min(unique_persons, unique_merchants)*100:.3f}%)")
    print(f"  Shared first-name tokens:   {len(token_overlap):>9,}")
    print(f"  Jaccard similarity:         {jaccard:>8.4f}%")
    print()

    if len(ci_overlap) == 0:
        print("  [PERFECT] Zero overlap between datasets!")
    elif len(ci_overlap) < 50:
        print("  [GOOD] Minimal overlap - minor cleanup needed")
    elif len(ci_overlap) < 500:
        print("  [WARNING] Moderate overlap - should investigate")
    else:
        print("  [CRITICAL] High overlap - datasets need deduplication")


if __name__ == "__main__":
    main()
