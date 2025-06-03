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
    print(congress_feature_importance)
    # Example analysis: Get top features for each congress (averaging across seeds)
    congress_nums = set()
    for key in congress_feature_importance.keys():
        congress_num = key.split('_')[0]
        congress_nums.add(congress_num)
    
    print(f"Congress numbers found: {sorted(congress_nums)}")
    
    return congress_feature_importance

def get_top_discriminative_terms(feature_importance_dict, top_n=20):
    """
    Get top N most discriminative terms (both positive and negative coefficients)
    """
    # Sort by absolute value of coefficients
    sorted_features = sorted(feature_importance_dict.items(), 
                        key=lambda x: abs(x[1]), reverse=True)
    return sorted_features[:top_n]

def create_evolution_dataframe(congress_top_terms, top_n_track=20):
    """
    Create a dataframe showing how top terms evolve over time
    """
    all_terms = set()
    for terms_list in congress_top_terms.values():
        all_terms.update([term for term, coef in terms_list[:top_n_track]])
    
    evolution_data = []
    for congress_num in sorted(congress_top_terms.keys()):
        term_ranks = {term: rank+1 for rank, (term, coef) in 
                    enumerate(congress_top_terms[congress_num])}
        
        for term in all_terms:
            evolution_data.append({
                'congress': congress_num,
                'term': term,
                'rank': term_ranks.get(term, None),  # None if not in top N
                'coefficient': feature_importance[congress_num].get(term, 0)
            })
    
    return pd.DataFrame(evolution_data)

# Run the analysis
if __name__ == "__main__":
    feature_importance = load_and_analyze_feature_importance()
    
    # Extract top terms for each congress
    congress_top_terms = {}
    for congress_num, importance_dict in feature_importance.items():
        congress_top_terms[congress_num] = get_top_discriminative_terms(importance_dict)
        evolution_df = create_evolution_dataframe(congress_top_terms)
    
#print(congress_top_terms)
    