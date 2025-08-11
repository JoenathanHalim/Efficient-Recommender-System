# Efficient-Recommender-System
Neural collaborative filtering that learns binary hash codes for users and items to enable fast, memory-efficient top-K recommendation via Hamming-distance search.

## Quick Start
The code targets a legacy TensorFlow 0.x stack. Run in an isolated env.

```bash
git clone https://github.com/SenticNet/recommendation-system.git
cd recommendation-system

# Example legacy env (adjust versions for your OS)
conda create -n dch-legacy python=3.5 -y
conda activate dch-legacy
pip install "tensorflow<1.0" numpy scipy scikit-learn

# Train (see script for flags/defaults)
python DCH.py

# Evaluate saved predictions
python evaluation.py
```

## Poster
[View the PDF](https://github.com/JoenathanHalim/Efficient-Recommender-System/blob/main/Efficient%20Recommender%20System.pdf)
