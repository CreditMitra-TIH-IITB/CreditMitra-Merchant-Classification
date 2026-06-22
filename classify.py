#!/usr/bin/env python
"""
Merchant Classifier — CLI Tool
==================================
Classify UPI names as person or merchant.

Usage:
    # Single name
    python classify.py "Swiggy Instamart"
    
    # Multiple names
    python classify.py "Rajesh Kumar" "HDFC Bank" "Priya Sharma"
    
    # CSV batch mode
    python classify.py --input names.csv --output results.csv
    
    # Interactive mode
    python classify.py --interactive
    
    # JSON output
    python classify.py --json "Swiggy"
    
    # Use CPU only
    python classify.py --cpu "Swiggy"
"""

import argparse
import csv
import json
import sys
import time
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_parser():
    parser = argparse.ArgumentParser(
        description="Classify UPI names as person or merchant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify.py "Swiggy Instamart"
  python classify.py "Rajesh Kumar" "HDFC Bank" "Zomato"
  python classify.py --input names.csv --output results.csv
  python classify.py --interactive
  python classify.py --json "Swiggy"
        """,
    )

    parser.add_argument(
        "names",
        nargs="*",
        help="Name(s) to classify",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input CSV file (must have a 'name' column)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output CSV file for batch results",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode — type names one at a time",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (don't use GPU)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to models directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )

    return parser


def print_result(result, json_mode=False):
    """Pretty-print a single result."""
    if json_mode:
        print(json.dumps(result, ensure_ascii=False))
    else:
        label = result["label"].upper()
        conf = result["confidence"]
        name = result.get("name", "")

        conf_bar = "#" * int(conf * 20) + "." * (20 - int(conf * 20))
        print(f"  {name:<30} -> {label:>10}  [{conf_bar}] {conf:.4f}")


def run_interactive(clf, json_mode=False):
    """Interactive classification mode."""
    print("\n  Interactive mode — type a name and press Enter (Ctrl+C to quit)\n")

    while True:
        try:
            name = input("  Name: ").strip()
            if not name:
                continue

            start = time.time()
            result = clf.classify(name)
            elapsed = time.time() - start

            print_result(result, json_mode)
            print(f"  ({elapsed*1000:.1f}ms)\n")

        except KeyboardInterrupt:
            print("\n\n  Goodbye!")
            break
        except EOFError:
            break


def run_batch(clf, input_file, output_file, json_mode=False):
    """Batch classification from CSV."""
    # Read input
    names = []
    extra_cols = {}

    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Find the name column (case-insensitive)
        name_col = None
        for col in fieldnames:
            if col.lower() in ("name", "names", "text", "input"):
                name_col = col
                break

        if name_col is None:
            # Use first column
            name_col = fieldnames[0] if fieldnames else "name"
            print(f"  Warning: No 'name' column found, using '{name_col}'")

        for row in reader:
            names.append(row[name_col])
            for col in fieldnames:
                if col != name_col:
                    extra_cols.setdefault(col, []).append(row[col])

    print(f"  Loaded {len(names):,} names from {input_file}")

    # Classify
    start = time.time()
    results = clf.classify_batch(names)
    elapsed = time.time() - start

    print(f"  Classified in {elapsed:.2f}s ({len(names)/elapsed:.0f} names/sec)")

    # Write output
    if output_file:
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            out_fields = ["name", "label", "confidence", "p_merchant"]
            out_fields += [c for c in extra_cols.keys()]
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()

            for i, result in enumerate(results):
                row = {
                    "name": result["name"],
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "p_merchant": result["p_merchant"],
                }
                for col, values in extra_cols.items():
                    row[col] = values[i]
                writer.writerow(row)

        print(f"  Results saved to {output_file}")
    else:
        # Print to stdout
        for result in results:
            print_result(result, json_mode)

    # Summary
    n_merchant = sum(1 for r in results if r["label"] == "merchant")
    n_person = len(results) - n_merchant
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    print(f"\n  Summary: {n_person:,} persons, {n_merchant:,} merchants (avg confidence: {avg_conf:.4f})")


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Validate args
    if not args.names and not args.input and not args.interactive:
        parser.print_help()
        print("\n  Error: Provide name(s), --input file, or --interactive")
        sys.exit(1)

    # Initialize classifier
    print("  Loading classifier...")
    from classifier import MerchantClassifier
    clf = MerchantClassifier(
        models_dir=args.models_dir,
        device="cpu" if args.cpu else "auto",
        embedding_batch_size=args.batch_size,
    )

    # Run appropriate mode
    if args.interactive:
        run_interactive(clf, args.json)

    elif args.input:
        run_batch(clf, args.input, args.output, args.json)

    elif args.names:
        # Classify provided names
        results = clf.classify_batch(args.names)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print()
            for result in results:
                print_result(result)
            print()


if __name__ == "__main__":
    main()
