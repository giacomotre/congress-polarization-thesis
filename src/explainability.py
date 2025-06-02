import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Load the saved feature importance
def load_and_analyze_feature_importance():
    # Load the dictionary
    with open("feature_importance/congress_feature_importance_svm.pkl", 'rb') as f:
        congress_feature_importance = pickle.load(f)
    
    print(f"Loaded feature importance for {len(congress_feature_importance)} congress-seed combinations")
    
    # Example analysis: Get top features for each congress (averaging across seeds)
    congress_nums = set()
    for key in congress_feature_importance.keys():
        congress_num = key.split('_')[0]
        congress_nums.add(congress_num)
    
    print(f"Congress numbers found: {sorted(congress_nums)}")
    
    return congress_feature_importance

# Run the analysis
if __name__ == "__main__":
    feature_importance = load_and_analyze_feature_importance()
    
    # Now you can run all your feature importance analysis here
    # without re-running the expensive model training