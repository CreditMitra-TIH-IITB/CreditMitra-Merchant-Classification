"""
UPI Name Augmentation Script
==============================
Takes clean Indian names and produces realistic UPI-style corrupted variants.

Real-world UPI narration issues modeled:
1.  TRUNCATION        - Bank systems cut names at 20-30 chars
2.  CHARACTER DROP     - Random chars missing (OCR/encoding issues)
3.  VOWEL DROP         - Vowels stripped ("Ramesh" -> "Rmsh")
4.  DOUBLE CHAR MERGE  - Repeated letters collapsed ("Chatterjee" -> "Chaterjee")
5.  SPACE ISSUES       - Extra spaces, missing spaces, underscores
6.  CASE NOISE         - Random capitalization ("rAmEsH sHaRmA")
7.  DOT/PERIOD INSERT  - "R.Sharma", "Ramesh.S"
8.  COMMON TYPOS       - Adjacent key swaps, phonetic substitutions
9.  PREFIX NOISE       - "UPI-", "Mr ", "MR " prepended
10. SUFFIX NOISE       - Numbers, codes appended ("RAMESH123", "Sharma-001")
11. TRANSLITERATION    - Common spelling variants ("Shukla"/"Shukl", "ee"/"i")
12. PARTIAL NAME       - Only first few chars visible ("RAME...")

Each augmented row keeps:
- The original name (for reference)
- The augmented name
- Which corruption was applied
- Still labeled as PERSON

Output: data/indian_person_names_augmented.csv
"""

import csv
import random
import os
import string


# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def truncate_name(name: str) -> str:
    """Simulate bank truncation at 15-25 characters."""
    max_len = random.randint(12, 25)
    if len(name) <= max_len:
        return name
    return name[:max_len].rstrip()


def truncate_with_ellipsis(name: str) -> str:
    """Truncate and add ... or . like some banks do."""
    max_len = random.randint(10, 20)
    if len(name) <= max_len:
        return name
    suffix = random.choice(["...", "..", ".", ""])
    return name[:max_len].rstrip() + suffix


def drop_random_chars(name: str) -> str:
    """Randomly remove 1-2 characters (simulating encoding/transmission errors)."""
    if len(name) < 4:
        return name
    chars = list(name)
    num_drops = random.randint(1, min(2, len(chars) - 3))
    # Don't drop spaces — that's a different corruption
    droppable = [i for i, c in enumerate(chars) if c != ' ']
    if len(droppable) < 2:
        return name
    indices = random.sample(droppable, num_drops)
    return ''.join(c for i, c in enumerate(chars) if i not in indices)


def drop_vowels(name: str) -> str:
    """Remove some vowels (common in SMS/abbreviated systems)."""
    vowels = set('aeiouAEIOU')
    result = []
    for i, c in enumerate(name):
        # Keep first char and spaces always
        if i == 0 or c == ' ' or c not in vowels:
            result.append(c)
        elif random.random() < 0.6:  # Drop 60% of vowels
            continue
        else:
            result.append(c)
    return ''.join(result)


def merge_double_chars(name: str) -> str:
    """Collapse repeated letters: 'Chatterjee' -> 'Chaterjee'."""
    if len(name) < 2:
        return name
    result = [name[0]]
    for i in range(1, len(name)):
        if name[i].lower() == name[i-1].lower() and name[i].isalpha():
            if random.random() < 0.7:  # 70% chance to merge
                continue
        result.append(name[i])
    return ''.join(result)


def add_extra_spaces(name: str) -> str:
    """Add random extra spaces between words."""
    parts = name.split()
    if len(parts) < 2:
        return name
    result = parts[0]
    for p in parts[1:]:
        spaces = ' ' * random.randint(2, 4)
        result += spaces + p
    return result


def remove_spaces(name: str) -> str:
    """Remove spaces between words: 'Ramesh Sharma' -> 'RameshSharma'."""
    return name.replace(' ', '')


def replace_space_with_char(name: str) -> str:
    """Replace spaces with underscores, dots, or dashes."""
    replacement = random.choice(['_', '.', '-', '/'])
    return name.replace(' ', replacement)


def random_case_noise(name: str) -> str:
    """Random capitalization mess."""
    return ''.join(
        c.upper() if random.random() < 0.4 else c.lower()
        for c in name
    )


def add_dots(name: str) -> str:
    """Add dots after initials or between parts: 'R.Sharma', 'R. K. Sharma'."""
    parts = name.split()
    if len(parts) < 2:
        return name
    result = []
    for p in parts:
        if len(p) == 1:
            result.append(p + '.')
        elif random.random() < 0.3:
            result.append(p + '.')
        else:
            result.append(p)
    return ' '.join(result)


def add_prefix_noise(name: str) -> str:
    """Add UPI/bank-style prefixes."""
    prefix = random.choice([
        "UPI-", "UPI/", "Mr ", "MR ", "Mrs ", "MRS ", "Shri ",
        "SHRI ", "Smt ", "SMT ", "Dr ", "DR ", "Sri ",
        "NEFT-", "IMPS-", "P2P-",
    ])
    return prefix + name


def add_suffix_noise(name: str) -> str:
    """Add numeric codes or identifiers at the end."""
    suffix = random.choice([
        str(random.randint(1, 999)),
        "-" + str(random.randint(100, 999)),
        "/" + str(random.randint(10, 99)),
        " " + ''.join(random.choices(string.ascii_uppercase, k=3)),
        "-" + ''.join(random.choices(string.digits, k=4)),
        " AC" + str(random.randint(1000, 9999)),
    ])
    return name + suffix


def common_typos(name: str) -> str:
    """Swap adjacent characters (common typing errors)."""
    if len(name) < 4:
        return name
    chars = list(name)
    # Find swappable positions (not spaces)
    positions = [i for i in range(len(chars) - 1)
                 if chars[i] != ' ' and chars[i+1] != ' ']
    if not positions:
        return name
    pos = random.choice(positions)
    chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
    return ''.join(chars)


def phonetic_substitution(name: str) -> str:
    """Common phonetic/transliteration variations in Indian names."""
    substitutions = [
        ('sh', 's'), ('sh', 'sch'), ('ph', 'f'), ('th', 't'),
        ('ee', 'i'), ('oo', 'u'), ('aa', 'a'), ('ii', 'i'),
        ('v', 'w'), ('w', 'v'), ('b', 'v'), ('z', 'j'),
        ('ch', 'c'), ('kh', 'k'), ('gh', 'g'), ('dh', 'd'),
        ('bh', 'b'), ('jh', 'j'), ('ks', 'x'), ('qu', 'k'),
        ('ey', 'ay'), ('ai', 'e'), ('au', 'o'), ('ou', 'u'),
    ]
    result = name.lower()
    # Apply 1-2 random substitutions
    applied = 0
    random.shuffle(substitutions)
    for old, new in substitutions:
        if old in result and applied < 2:
            result = result.replace(old, new, 1)
            applied += 1
    # Restore original casing pattern roughly
    final = []
    for i, c in enumerate(result):
        if i < len(name) and name[i].isupper():
            final.append(c.upper())
        else:
            final.append(c)
    return ''.join(final)


def partial_name_with_mask(name: str) -> str:
    """Show only first part, mask the rest: 'RAME***', 'Shar...'."""
    if len(name) < 5:
        return name
    visible = random.randint(3, min(8, len(name) - 2))
    mask = random.choice(['***', '...', 'XXX', '***', '..'])
    return name[:visible] + mask


def strip_last_name(name: str) -> str:
    """Only keep first name from a multi-part name."""
    parts = name.split()
    if len(parts) >= 2:
        return parts[0]
    return name


def strip_first_name(name: str) -> str:
    """Only keep last name from a multi-part name."""
    parts = name.split()
    if len(parts) >= 2:
        return parts[-1]
    return name


def first_n_chars_of_each_word(name: str) -> str:
    """Abbreviate each word: 'Ramesh Kumar Sharma' -> 'Ram Kum Sha'."""
    parts = name.split()
    n = random.randint(2, 4)
    return ' '.join(p[:n] for p in parts)


def insert_random_char(name: str) -> str:
    """Insert a random character (keyboard noise)."""
    if len(name) < 3:
        return name
    pos = random.randint(1, len(name) - 1)
    char = random.choice(string.ascii_lowercase)
    return name[:pos] + char + name[pos:]


def repeat_a_char(name: str) -> str:
    """Accidentally type a letter twice: 'Ramesh' -> 'Rammesh'."""
    if len(name) < 3:
        return name
    positions = [i for i, c in enumerate(name) if c.isalpha()]
    if not positions:
        return name
    pos = random.choice(positions)
    return name[:pos] + name[pos] + name[pos:]


# ============================================================================
# AUGMENTATION PIPELINE
# ============================================================================

# Each augmentation: (function, weight, name)
AUGMENTATIONS = [
    (truncate_name,                3, "truncated"),
    (truncate_with_ellipsis,       2, "truncated_ellipsis"),
    (drop_random_chars,            3, "char_drop"),
    (drop_vowels,                  2, "vowel_drop"),
    (merge_double_chars,           2, "double_merge"),
    (add_extra_spaces,             2, "extra_spaces"),
    (remove_spaces,                2, "no_spaces"),
    (replace_space_with_char,      2, "space_replaced"),
    (random_case_noise,            2, "case_noise"),
    (add_dots,                     2, "dots_added"),
    (add_prefix_noise,             2, "prefix_noise"),
    (add_suffix_noise,             2, "suffix_noise"),
    (common_typos,                 3, "typo_swap"),
    (phonetic_substitution,        3, "phonetic_sub"),
    (partial_name_with_mask,       2, "partial_masked"),
    (strip_last_name,              2, "first_name_only"),
    (strip_first_name,             2, "last_name_only"),
    (first_n_chars_of_each_word,   2, "abbreviated"),
    (insert_random_char,           1, "char_insert"),
    (repeat_a_char,                1, "char_repeat"),
]


def augment_name(name: str, num_augments: int = 2) -> list[dict]:
    """
    Apply random augmentations to a name.

    Args:
        name: The clean name
        num_augments: Number of augmented variants to generate per name

    Returns:
        List of dicts with augmented name and augmentation type
    """
    results = []

    # Build weighted pool
    pool = []
    for func, weight, aug_name in AUGMENTATIONS:
        pool.extend([(func, aug_name)] * weight)

    for _ in range(num_augments):
        func, aug_type = random.choice(pool)
        augmented = func(name)

        # Skip if augmentation produced the same string
        if augmented.lower().strip() == name.lower().strip():
            # Try another one
            func, aug_type = random.choice(pool)
            augmented = func(name)

        if augmented and augmented.strip():
            results.append({
                "augmented_name": augmented.strip(),
                "augmentation": aug_type,
            })

    # Occasionally apply compound corruption (2 augmentations stacked)
    if random.random() < 0.15:  # 15% chance
        func1, aug1 = random.choice(pool)
        func2, aug2 = random.choice(pool)
        compound = func2(func1(name))
        if compound and compound.strip():
            results.append({
                "augmented_name": compound.strip(),
                "augmentation": f"{aug1}+{aug2}",
            })

    return results


def augment_dataset(
    input_path: str,
    output_path: str,
    augments_per_name: int = 2,
    seed: int = 123,
):
    """
    Read the clean dataset, augment it, and write combined output.

    The output contains:
    - ALL original clean names (with augmentation="original")
    - Augmented variants for each name
    """
    random.seed(seed)

    # Read clean dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        clean_rows = list(reader)

    print(f"Read {len(clean_rows)} clean names from {input_path}")

    # Generate augmented dataset
    output_rows = []
    seen = set()

    for row in clean_rows:
        original_name = row['name']

        # Keep original
        key = original_name.lower().strip()
        if key not in seen:
            seen.add(key)
            output_rows.append({
                'name': original_name,
                'label': 'PERSON',
                'gender': row.get('gender', ''),
                'region': row.get('region', ''),
                'format': row.get('format', ''),
                'augmentation': 'original',
                'original_name': original_name,
            })

        # Generate augmented variants
        augmented = augment_name(original_name, num_augments=augments_per_name)
        for aug in augmented:
            aug_key = aug['augmented_name'].lower().strip()
            if aug_key not in seen and aug_key:
                seen.add(aug_key)
                output_rows.append({
                    'name': aug['augmented_name'],
                    'label': 'PERSON',
                    'gender': row.get('gender', ''),
                    'region': row.get('region', ''),
                    'format': f"{row.get('format', '')}_{aug['augmentation']}",
                    'augmentation': aug['augmentation'],
                    'original_name': original_name,
                })

    # Shuffle
    random.shuffle(output_rows)

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ['name', 'label', 'gender', 'region', 'format', 'augmentation', 'original_name']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[OK] Saved {len(output_rows)} names to {output_path}")
    return output_rows


def print_augment_stats(rows: list[dict]):
    """Print augmentation statistics."""
    from collections import Counter

    print("\n" + "=" * 60)
    print("AUGMENTED DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal entries: {len(rows)}")

    # Original vs augmented
    original = sum(1 for r in rows if r['augmentation'] == 'original')
    augmented = len(rows) - original
    print(f"\n  Original (clean):  {original:6d}")
    print(f"  Augmented (noisy): {augmented:6d}")
    print(f"  Augmentation ratio: {augmented/original:.2f}x")

    # By augmentation type
    aug_counts = Counter(r['augmentation'] for r in rows if r['augmentation'] != 'original')
    print(f"\nAugmentation Types ({len(aug_counts)}):")
    for aug, count in sorted(aug_counts.items(), key=lambda x: -x[1]):
        print(f"  {aug:30s} -> {count:5d}")

    # Samples of each type
    print(f"\nSamples by augmentation type:")
    print("-" * 80)
    shown_types = set()
    for row in rows:
        aug = row['augmentation']
        if aug not in shown_types and aug != 'original':
            shown_types.add(aug)
            orig = row.get('original_name', '')
            noisy = row['name']
            print(f"  [{aug:25s}]  {orig:30s}  ->  {noisy}")
        if len(shown_types) >= 20:
            break

    # Overall sample
    print(f"\nRandom sample (20 entries):")
    print("-" * 80)
    sample = random.sample(rows, min(20, len(rows)))
    for row in sample:
        tag = "CLEAN" if row['augmentation'] == 'original' else "NOISY"
        print(f"  [{tag}] {row['name']:35s} | {row['augmentation']:20s} | {row['region']}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), "indian_person_names.csv")
    output_path = os.path.join(os.path.dirname(__file__), "indian_person_names_augmented.csv")

    rows = augment_dataset(
        input_path=input_path,
        output_path=output_path,
        augments_per_name=2,   # 2 augmented variants per clean name
        seed=123,
    )

    print_augment_stats(rows)
