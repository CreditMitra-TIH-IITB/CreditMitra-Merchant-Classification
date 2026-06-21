import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_mem / 1024**3:.1f} GB")

# Check other libs
try:
    import sentence_transformers
    print(f"sentence-transformers: {sentence_transformers.__version__}")
except ImportError:
    print("sentence-transformers: NOT INSTALLED")

try:
    import sklearn
    print(f"scikit-learn: {sklearn.__version__}")
except ImportError:
    print("scikit-learn: NOT INSTALLED")

try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError:
    print("numpy: NOT INSTALLED")

try:
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")
except ImportError:
    print("matplotlib: NOT INSTALLED")

try:
    import umap
    print(f"umap-learn: {umap.__version__}")
except ImportError:
    print("umap-learn: NOT INSTALLED")

try:
    import seaborn
    print(f"seaborn: {seaborn.__version__}")
except ImportError:
    print("seaborn: NOT INSTALLED")
