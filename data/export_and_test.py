"""
ONNX Export + Edge Case Testing + Unseen Data Validation
==========================================================
1. Export AttentionMLP to ONNX
2. Benchmark ONNX vs PyTorch inference speed
3. Test edge cases and adversarial examples
4. Generate unseen test data and validate

Usage:
    python data/export_and_test.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import AttentionMLP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "research", "figures")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PHASE 1: ONNX EXPORT
# ============================================================================

def export_onnx():
    """Export AttentionMLP to ONNX format."""
    print("=" * 70)
    print("PHASE 1: ONNX EXPORT")
    print("=" * 70)

    # Load PyTorch model
    model_path = os.path.join(MODELS_DIR, "pytorch_attentionmlp.pt")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    model = AttentionMLP(input_dim=1024, num_heads=8, dropout=0.0)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 1024)

    # Export
    onnx_path = os.path.join(MODELS_DIR, "attentionmlp.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["embedding"],
        output_names=["logit"],
        dynamic_axes={
            "embedding": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
    )

    onnx_size = os.path.getsize(onnx_path)
    pytorch_size = os.path.getsize(model_path)
    print(f"\n  ONNX saved: {onnx_path}")
    print(f"  ONNX size:    {onnx_size / 1024 / 1024:.2f} MB")
    print(f"  PyTorch size: {pytorch_size / 1024 / 1024:.2f} MB")
    print(f"  Compression:  {pytorch_size / onnx_size:.1f}x smaller")

    return onnx_path


# ============================================================================
# PHASE 2: BENCHMARK
# ============================================================================

def benchmark(onnx_path):
    """Benchmark ONNX vs PyTorch inference speed."""
    print("\n" + "=" * 70)
    print("PHASE 2: INFERENCE BENCHMARK")
    print("=" * 70)

    import onnxruntime as ort

    # Load models
    model_path = os.path.join(MODELS_DIR, "pytorch_attentionmlp.pt")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    pytorch_model = AttentionMLP(input_dim=1024, num_heads=8, dropout=0.0)
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.to(DEVICE)
    pytorch_model.eval()

    # ONNX session (CPU only — production use case)
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Benchmark single-sample inference (most realistic for production)
    test_sizes = [1, 10, 100, 1000]
    results = {}

    warmup_iters = 10
    bench_iters = 50

    for N in test_sizes:
        print(f"\n  Batch size: {N}")
        X = np.random.randn(N, 1024).astype(np.float32)

        # --- PyTorch (batched) ---
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = pytorch_model(torch.FloatTensor(X).to(DEVICE))

        times = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            with torch.no_grad():
                logits = pytorch_model(torch.FloatTensor(X).to(DEVICE))
                _ = torch.sigmoid(logits).cpu().numpy()
            times.append(time.perf_counter() - start)
        pt_time = np.median(times) * 1000
        print(f"    PyTorch ({DEVICE}): {pt_time:.2f}ms (median)")

        # --- ONNX CPU (single-sample loop for batch > 1) ---
        # ONNX model was exported with batch_size=1 due to attention reshape
        for _ in range(warmup_iters):
            _ = ort_session.run(None, {"embedding": X[:1]})

        times = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            onnx_results = []
            for i in range(N):
                out = ort_session.run(None, {"embedding": X[i:i+1]})
                onnx_results.append(out[0])
            times.append(time.perf_counter() - start)
        onnx_cpu_time = np.median(times) * 1000
        print(f"    ONNX (CPU):       {onnx_cpu_time:.2f}ms (median, {N}x single-sample)")

        speedup = pt_time / onnx_cpu_time if onnx_cpu_time > 0 else 0
        print(f"    Speedup:          {speedup:.1f}x")

        results[N] = {
            "pytorch_ms": round(pt_time, 2),
            "onnx_cpu_ms": round(onnx_cpu_time, 2),
            "onnx_gpu_ms": None,
            "speedup_cpu": round(speedup, 2),
        }

    # Verify numerical equivalence (single sample)
    print("\n  Numerical equivalence check:")
    max_diffs = []
    for i in range(100):
        X_check = np.random.randn(1, 1024).astype(np.float32)
        with torch.no_grad():
            pt_out = torch.sigmoid(pytorch_model(torch.FloatTensor(X_check).to(DEVICE))).cpu().numpy()
        onnx_out_raw = ort_session.run(None, {"embedding": X_check})[0]
        onnx_out = 1.0 / (1.0 + np.exp(-onnx_out_raw))
        max_diffs.append(abs(pt_out.item() - onnx_out.item()))

    max_diff = max(max_diffs)
    avg_diff = sum(max_diffs) / len(max_diffs)
    print(f"    Max absolute difference: {max_diff:.8f}")
    print(f"    Avg absolute difference: {avg_diff:.8f}")
    print(f"    Numerically equivalent: {'YES' if max_diff < 1e-4 else 'NO'}")

    return results


# ============================================================================
# PHASE 3: EDGE CASES & ADVERSARIAL TESTING
# ============================================================================

def test_edge_cases():
    """Test the classifier on edge cases and adversarial examples."""
    print("\n" + "=" * 70)
    print("PHASE 3: EDGE CASE & ADVERSARIAL TESTING")
    print("=" * 70)

    from classifier import MerchantClassifier
    clf = MerchantClassifier(device="cuda" if torch.cuda.is_available() else "cpu")

    test_categories = {
        # ---- CLEAR PERSONS ----
        "Clear Person Names": {
            "expected": "person",
            "names": [
                "Rajesh Kumar",
                "Priya Sharma",
                "Mohammed Irfan",
                "Deepika Padukone",
                "Amit Patel",
                "Sneha Reddy",
                "Vikram Singh Chauhan",
                "Ananya Iyer",
                "Suresh Babu",
                "Fatima Begum",
            ]
        },
        # ---- CLEAR MERCHANTS ----
        "Clear Merchant Names": {
            "expected": "merchant",
            "names": [
                "Swiggy Instamart",
                "HDFC Bank",
                "Amazon Pay",
                "Flipkart Seller Hub",
                "Zomato Online Order",
                "Reliance Jio",
                "Paytm Payments Bank",
                "BigBasket Daily",
                "PhonePe Merchant",
                "Uber India",
            ]
        },
        # ---- AMBIGUOUS: Person names that look like merchants ----
        "Ambiguous (Person-like Merchants)": {
            "expected": "merchant",
            "names": [
                "Tanishq",          # Jewellery brand but sounds like a name
                "Lakme",            # Beauty brand but sounds like a name
                "Allen",            # Coaching institute but a name
                "Bata",             # Shoe brand
                "Raymond",          # Clothing brand but a name
                "Peter England",    # Clothing brand with person name
                "Louis Philippe",   # Brand with person name
                "Van Heusen",       # Brand with person name
                "Monte Carlo",      # Brand
                "Kalyan Jewellers", # Named after a person
            ]
        },
        # ---- AMBIGUOUS: Merchants that look like persons ----
        "Ambiguous (Merchant-like Persons)": {
            "expected": "person",
            "names": [
                "Tanishq Sharma",   # Person with brand-like first name
                "Raymond Singh",    # Person with brand-like first name
                "Allen Kumar",      # Person with brand-like first name
                "Uber Pandey",      # Unlikely but possible
                "Lakshmi Gold",     # Person or gold shop?
            ]
        },
        # ---- EDGE: Very short names ----
        "Edge: Very Short": {
            "expected": None,  # Could go either way
            "names": [
                "A",
                "AB",
                "Ram",
                "SBI",
                "HP",
                "LG",
                "Mi",
            ]
        },
        # ---- EDGE: Very long names ----
        "Edge: Very Long": {
            "expected": None,
            "names": [
                "Shri Venkateshwara Swamy Temple Trust Charitable Foundation",
                "Mohammed Abdul Rehman Khan Pathan",
                "National Stock Exchange of India Limited Mumbai Branch",
                "Dr Baba Saheb Ambedkar University Lucknow Uttar Pradesh",
            ]
        },
        # ---- EDGE: Numbers and codes ----
        "Edge: Numbers & Codes": {
            "expected": None,
            "names": [
                "UPI-1234567890",
                "NEFT-REF123",
                "IMPS-45678",
                "99acres",
                "Flat-202",
                "1MG",
                "5Paisa",
            ]
        },
        # ---- ADVERSARIAL: Mixed case ----
        "Adversarial: Mixed Case": {
            "expected": None,
            "names": [
                "sWiGgY",
                "RAJESH KUMAR",
                "rajesh kumar",
                "RaJeSh KuMaR",
                "flipkart",
                "FLIPKART",
                "fLiPkArT",
            ]
        },
        # ---- ADVERSARIAL: Truncated/Abbreviated ----
        "Adversarial: Truncated": {
            "expected": None,
            "names": [
                "Raj...",
                "Swig***",
                "HDF***",
                "Pr Sh",
                "A K Enterprises",
                "S K Traders",
                "M/s Sharma",
            ]
        },
        # ---- ADVERSARIAL: With UPI prefixes ----
        "Adversarial: UPI Prefixes": {
            "expected": None,
            "names": [
                "UPI-Rajesh Kumar",
                "P2P-Priya Sharma",
                "UPI-Swiggy",
                "NEFT-HDFC Bank",
                "IMPS-Deepak Singh",
                "P2M-Amazon Pay",
            ]
        },
        # ---- EDGE: Non-Indian names ----
        "Edge: Non-Indian Names": {
            "expected": "person",
            "names": [
                "John Smith",
                "Michael Johnson",
                "Sarah Williams",
                "Carlos Rodriguez",
                "Yuki Tanaka",
                "Ahmed Hassan",
            ]
        },
        # ---- EDGE: Empty/Whitespace ----
        "Edge: Empty & Special": {
            "expected": None,
            "names": [
                " ",
                "  ",
                ".",
                "-",
                "N/A",
                "NA",
                "SELF",
                "self",
            ]
        },
    }

    all_results = {}
    total_tests = 0
    total_correct = 0

    for category, config in test_categories.items():
        expected = config["expected"]
        names = config["names"]

        print(f"\n  [{category}] (expected: {expected or 'any'})")
        results = clf.classify_batch(names)

        category_results = []
        correct = 0

        for result in results:
            name = result["name"]
            label = result["label"]
            conf = result["confidence"]
            p_merchant = result["p_merchant"]

            if expected:
                is_correct = (label == expected)
                correct += int(is_correct)
                status = "OK" if is_correct else "WRONG"
            else:
                is_correct = None
                status = ""

            category_results.append({
                "name": name,
                "label": label,
                "confidence": conf,
                "p_merchant": p_merchant,
                "expected": expected,
                "correct": is_correct,
            })

            # Format output
            label_str = f"{label.upper():>10}"
            print(f"    {name:<55} {label_str} ({conf:.4f}) {status}")

        if expected:
            acc = correct / len(names) * 100
            total_correct += correct
            total_tests += len(names)
            print(f"    Score: {correct}/{len(names)} ({acc:.0f}%)")

        all_results[category] = category_results

    if total_tests > 0:
        overall_acc = total_correct / total_tests * 100
        print(f"\n  Overall (on categories with expected labels): {total_correct}/{total_tests} ({overall_acc:.1f}%)")

    return all_results


# ============================================================================
# PHASE 4: UNSEEN DATA GENERATION & TESTING
# ============================================================================

def test_unseen_data():
    """Generate completely new names and test the classifier."""
    print("\n" + "=" * 70)
    print("PHASE 4: UNSEEN DATA TESTING")
    print("=" * 70)

    from classifier import MerchantClassifier
    clf = MerchantClassifier(device="cuda" if torch.cuda.is_available() else "cpu")

    # Completely new person names (not in training data)
    unseen_persons = [
        # Full names
        "Aarav Mehra", "Kavya Krishnan", "Rohan Deshmukh", "Ishita Banerjee",
        "Yash Agrawal", "Diya Nair", "Arjun Malhotra", "Tara Chowdhury",
        "Vivaan Saxena", "Aisha Khan", "Reyansh Dubey", "Myra Bhat",
        "Kabir Rana", "Anvi Jha", "Vihaan Pillai", "Saanvi Srinivasan",
        # Short/informal
        "Ritu", "Karan", "Neha", "Arun", "Pooja", "Sanjay",
        # With initials
        "R. K. Mishra", "S. Venkataraman", "P. Raghunathan",
        # South Indian
        "Venkatesh Prasad", "Lakshmi Narasimhan", "Radhakrishnan Pillai",
        # Muslim names
        "Zainab Fatima", "Imran Hashmi", "Ayesha Siddiqui",
        # With prefixes
        "Dr Suresh Kumar", "Smt Kamla Devi", "Shri Ram Prasad",
    ]

    # Completely new merchant names (not in training data)
    unseen_merchants = [
        # Tech
        "Groww App", "Zerodha Kite", "PharmEasy", "Lenskart Online",
        "PolicyBazaar", "CarDekho", "Nykaa Fashion", "Meesho Seller",
        # Food
        "Biryani By Kilo", "Theobroma Bakery", "Barbeque Nation",
        "Haldiram Nagpur", "MTR Foods", "ID Fresh Food",
        # Services
        "Urban Company", "PorterApp Delivery", "Dunzo Daily",
        "Practo Health", "Cult.fit Premium",
        # Local shops
        "Sharma General Store", "Patel Electronics", "Singh Auto Parts",
        "Gupta Medical Store", "Khan Tailors",
        # Utilities
        "BESCOM Electricity", "Mahanagar Gas Ltd", "BSNL Recharge",
        "Jio Fiber", "Tata Power Solar",
        # Banking
        "ICICI Prudential", "SBI Life Insurance", "Bajaj Finance EMI",
        "CRED Payments", "Google Pay Merchant",
    ]

    # Test persons
    print("\n  Testing UNSEEN Person Names:")
    person_results = clf.classify_batch(unseen_persons)
    person_correct = sum(1 for r in person_results if r["label"] == "person")
    person_total = len(person_results)

    for r in person_results:
        status = "OK" if r["label"] == "person" else "WRONG"
        print(f"    {r['name']:<40} {r['label'].upper():>10} ({r['confidence']:.4f}) {status}")

    print(f"\n    Person accuracy: {person_correct}/{person_total} ({person_correct/person_total*100:.1f}%)")

    # Test merchants
    print("\n  Testing UNSEEN Merchant Names:")
    merchant_results = clf.classify_batch(unseen_merchants)
    merchant_correct = sum(1 for r in merchant_results if r["label"] == "merchant")
    merchant_total = len(merchant_results)

    for r in merchant_results:
        status = "OK" if r["label"] == "merchant" else "WRONG"
        print(f"    {r['name']:<40} {r['label'].upper():>10} ({r['confidence']:.4f}) {status}")

    print(f"\n    Merchant accuracy: {merchant_correct}/{merchant_total} ({merchant_correct/merchant_total*100:.1f}%)")

    # Overall
    total = person_total + merchant_total
    correct = person_correct + merchant_correct
    print(f"\n  UNSEEN DATA OVERALL: {correct}/{total} ({correct/total*100:.1f}%)")

    return {
        "person_accuracy": person_correct / person_total,
        "merchant_accuracy": merchant_correct / merchant_total,
        "overall_accuracy": correct / total,
        "person_errors": [r for r in person_results if r["label"] != "person"],
        "merchant_errors": [r for r in merchant_results if r["label"] != "merchant"],
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Phase 1: ONNX Export
    onnx_path = export_onnx()

    # Phase 2: Benchmark
    try:
        import onnxruntime
        bench_results = benchmark(onnx_path)
    except ImportError:
        print("\n  onnxruntime not installed, skipping benchmark")
        print("  Install with: pip install onnxruntime-gpu")
        bench_results = None

    # Phase 3: Edge Cases
    edge_results = test_edge_cases()

    # Phase 4: Unseen Data
    unseen_results = test_unseen_data()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  ONNX Model: {os.path.join(MODELS_DIR, 'attentionmlp.onnx')}")

    if bench_results:
        print(f"\n  Benchmark (median latency):")
        for N, r in bench_results.items():
            print(f"    Batch {N:>5}: PyTorch={r['pytorch_ms']:.2f}ms, "
                  f"ONNX CPU={r['onnx_cpu_ms']:.2f}ms "
                  f"({'%.1fx' % r['speedup_cpu']} speedup)")

    print(f"\n  Unseen Data:")
    print(f"    Person accuracy:   {unseen_results['person_accuracy']*100:.1f}%")
    print(f"    Merchant accuracy: {unseen_results['merchant_accuracy']*100:.1f}%")
    print(f"    Overall:           {unseen_results['overall_accuracy']*100:.1f}%")

    if unseen_results['person_errors']:
        print(f"\n  Person errors ({len(unseen_results['person_errors'])}):")
        for e in unseen_results['person_errors']:
            print(f"    '{e['name']}' classified as {e['label']} ({e['confidence']:.4f})")

    if unseen_results['merchant_errors']:
        print(f"\n  Merchant errors ({len(unseen_results['merchant_errors'])}):")
        for e in unseen_results['merchant_errors']:
            print(f"    '{e['name']}' classified as {e['label']} ({e['confidence']:.4f})")

    # Save all results
    summary = {
        "onnx_export": {
            "path": os.path.join(MODELS_DIR, "attentionmlp.onnx"),
            "size_mb": os.path.getsize(os.path.join(MODELS_DIR, "attentionmlp.onnx")) / 1024 / 1024,
        },
        "benchmark": bench_results,
        "unseen_data": {
            "person_accuracy": unseen_results["person_accuracy"],
            "merchant_accuracy": unseen_results["merchant_accuracy"],
            "overall_accuracy": unseen_results["overall_accuracy"],
        },
    }

    with open(os.path.join(MODELS_DIR, "onnx_test_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Results saved to models/onnx_test_results.json")
    print("\nDONE!")


if __name__ == "__main__":
    main()
