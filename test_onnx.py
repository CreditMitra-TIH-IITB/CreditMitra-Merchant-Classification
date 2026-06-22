"""Quick test of the standalone ONNX classifier."""
from onnx_classifier import ONNXMerchantClassifier

clf = ONNXMerchantClassifier()

names = [
    "Swiggy Instamart", "Rajesh Kumar", "HDFC Bank",
    "Priya Sharma", "Amazon Pay", "Deepak Singh",
    "Zomato", "Flipkart", "Rahul Verma", "Ola Cabs",
]

results = clf.classify_batch(names)
for r in results:
    name = r["name"]
    label = r["label"].upper()
    conf = r["confidence"]
    print(f"  {name:<25} -> {label:>10} ({conf:.4f})")
