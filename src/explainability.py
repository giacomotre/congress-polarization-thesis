import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class PolarizationAnalysisPipeline:
    def __init__(self, pickle_path):
        self.pickle_path = pickle_path
        self.raw_data = None
        self.aggregated_data = None
        self.polarization_analysis = None
    
    def load_pickle(self):
        """Step 1: Load the pickle file"""
        print("Loading pickle file...")
        with open(self.pickle_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        
        congress_nums = set()
        seeds = set()
        for key in self.raw_data.keys():
            congress_num, seed = key.split('_')
            congress_nums.add(int(congress_num))
            seeds.add(int(seed))
        
        print(f"Loaded data for {len(congress_nums)} congress sessions: {sorted(congress_nums)}")
        print(f"Using {len(seeds)} seeds: {sorted(seeds)}")
        
        return self.raw_data
    
    def aggregate_all_terms(self):
        """Step 2: Aggregate ALL terms across seeds for each congress"""
        print("Aggregating coefficients across seeds...")
        
        congress_term_coefficients = defaultdict(lambda: defaultdict(list))
        
        for key, term_coeffs in self.raw_data.items():
            congress_num, seed = key.split('_')
            congress_num = int(congress_num)
            
            for term, coefficient in term_coeffs.items():
                congress_term_coefficients[congress_num][term].append(coefficient)
        
        # Calculate aggregated statistics
        self.aggregated_data = {}
        for congress_num, terms_dict in congress_term_coefficients.items():
            self.aggregated_data[congress_num] = {}
            
            for term, coeff_list in terms_dict.items():
                mean_coeff = np.mean(coeff_list)
                self.aggregated_data[congress_num][term] = {
                    'mean_coefficient': mean_coeff,
                    'abs_coefficient': abs(mean_coeff),
                    'std_coefficient': np.std(coeff_list),
                    'n_seeds': len(coeff_list)
                }
        
        return self.aggregated_data
    
    def analyze_partisan_distinctions(self, top_n=20):
        """Step 3: Analyze top Republican vs Democrat distinguishing terms"""
        print(f"\nAnalyzing top {top_n} partisan distinguishing terms per congress...")
        
        self.polarization_analysis = {}
        
        for congress_num, terms_data in self.aggregated_data.items():
            # Sort ALL terms by absolute coefficient (most discriminative)
            sorted_terms = sorted(terms_data.items(), 
                                key=lambda x: x[1]['abs_coefficient'], 
                                reverse=True)
            
            # Get top Republican terms (positive coefficients)
            republican_terms = []
            democrat_terms = []
            
            for term, data in sorted_terms:
                if data['mean_coefficient'] > 0 and len(republican_terms) < top_n:
                    republican_terms.append({
                        'term': term,
                        'coefficient': data['mean_coefficient'],
                        'abs_coefficient': data['abs_coefficient'],
                        'std': data['std_coefficient']
                    })
                elif data['mean_coefficient'] < 0 and len(democrat_terms) < top_n:
                    democrat_terms.append({
                        'term': term,
                        'coefficient': data['mean_coefficient'],
                        'abs_coefficient': data['abs_coefficient'],
                        'std': data['std_coefficient']
                    })
                
                # Stop when we have enough of both
                if len(republican_terms) >= top_n and len(democrat_terms) >= top_n:
                    break
            
            self.polarization_analysis[congress_num] = {
                'republican_terms': republican_terms,
                'democrat_terms': democrat_terms
            }
        
        return self.polarization_analysis
    
    def print_partisan_comparison(self):
        """Print detailed partisan comparison for each congress"""
        if not self.polarization_analysis:
            print("Run analyze_partisan_distinctions() first!")
            return
        
        print("\n" + "="*80)
        print("PARTISAN LANGUAGE DISTINCTION ANALYSIS")
        print("="*80)
        
        for congress_num in sorted(self.polarization_analysis.keys()):
            data = self.polarization_analysis[congress_num]
            
            print(f"\nCONGRESS {congress_num}")
            print("-" * 50)
            
            print(f"\nTOP REPUBLICAN-DISTINGUISHING TERMS:")
            print("  (Positive coefficients - algorithm associates these with Republicans)")
            for i, term_data in enumerate(data['republican_terms'][:10], 1):
                print(f"  {i:2d}. {term_data['term']:<15} (coeff: {term_data['coefficient']:+.3f})")
            
            print(f"\nTOP DEMOCRAT-DISTINGUISHING TERMS:")
            print("  (Negative coefficients - algorithm associates these with Democrats)")
            for i, term_data in enumerate(data['democrat_terms'][:10], 1):
                print(f"  {i:2d}. {term_data['term']:<15} (coeff: {term_data['coefficient']:+.3f})")
    
    def create_polarization_evolution_df(self):
        """Create dataframe showing polarization evolution over time"""
        if not self.polarization_analysis:
            print("Run analyze_partisan_distinctions() first!")
            return None
        
        evolution_data = []
        
        for congress_num in sorted(self.polarization_analysis.keys()):
            data = self.polarization_analysis[congress_num]
            
            evolution_data.append({
                'congress': congress_num,
                'strongest_republican_term': data['republican_terms'][0]['term'] if data['republican_terms'] else None,
                'strongest_democrat_term': data['democrat_terms'][0]['term'] if data['democrat_terms'] else None
            })
        
        return pd.DataFrame(evolution_data)
    
    def compare_congress_polarization(self, congress1, congress2):
        """Compare polarization between two specific congresses"""
        if not self.polarization_analysis:
            print("Run analyze_partisan_distinctions() first!")
            return
        
        if congress1 not in self.polarization_analysis or congress2 not in self.polarization_analysis:
            print(f"Congress {congress1} or {congress2} not found in data")
            return
        
        data1 = self.polarization_analysis[congress1]
        data2 = self.polarization_analysis[congress2]
        
        print(f"\nCOMPARING CONGRESS {congress1} vs CONGRESS {congress2}")
        print("="*60)
        
        # Top terms comparison
        print(f"\nTop Republican Terms Comparison:")
        rep1_terms = [t['term'] for t in data1['republican_terms'][:5]]
        rep2_terms = [t['term'] for t in data2['republican_terms'][:5]]
        
        print(f"  Congress {congress1}: {', '.join(rep1_terms)}")
        print(f"  Congress {congress2}: {', '.join(rep2_terms)}")
        
        common_rep = set(rep1_terms) & set(rep2_terms)
        if common_rep:
            print(f"  Common terms: {', '.join(common_rep)}")
        
        print(f"\nTop Democrat Terms Comparison:")
        dem1_terms = [t['term'] for t in data1['democrat_terms'][:5]]
        dem2_terms = [t['term'] for t in data2['democrat_terms'][:5]]
        
        print(f"  Congress {congress1}: {', '.join(dem1_terms)}")
        print(f"  Congress {congress2}: {', '.join(dem2_terms)}")
        
        common_dem = set(dem1_terms) & set(dem2_terms)
        if common_dem:
            print(f"  Common terms: {', '.join(common_dem)}")
    
    def get_polarization_trend_summary(self):
        """Get summary of polarization trends"""
        evolution_df = self.create_polarization_evolution_df()
        
        if len(evolution_df) < 2:
            print("Need at least 2 congress sessions to analyze trends")
            return
        
        print("\nPOLARIZATION TREND SUMMARY")
        print("="*50)
        
        first_congress = evolution_df.iloc[0]
        last_congress = evolution_df.iloc[-1]
        
        print(f"Period: Congress {first_congress['congress']} → Congress {last_congress['congress']}")
        print(f"Analysis shows evolution in partisan language patterns over time")
        
        return evolution_df
    
    def run_polarization_pipeline(self, top_n=20):
        """Run the complete polarization analysis pipeline"""
        print("RUNNING CONGRESSIONAL POLARIZATION ANALYSIS PIPELINE")
        print("="*70)
        
        # Load and aggregate data
        self.load_pickle()
        self.aggregate_all_terms()
        
        # Analyze partisan distinctions
        self.analyze_partisan_distinctions(top_n)
        
        # Print detailed analysis
        self.print_partisan_comparison()
        
        # Get trend summary
        evolution_df = self.get_polarization_trend_summary()
        
        return self.polarization_analysis, evolution_df

# Usage example
def main():
    pipeline = PolarizationAnalysisPipeline("feature_importance/congress_feature_importance_svm.pkl")
    
    # Run full analysis
    analysis, evolution_df = pipeline.run_polarization_pipeline(top_n=20)
    
    # Optional: Compare specific congresses
    if len(evolution_df) >= 2:
        congress_list = sorted(evolution_df['congress'].tolist())
        pipeline.compare_congress_polarization(congress_list[0], congress_list[-1])
    
    return pipeline, analysis, evolution_df

if __name__ == "__main__":
    pipeline, analysis, evolution_df = main()