import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

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
    
    def analyze_partisan_distinctions(self, top_n=15):
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
            for i, term_data in enumerate(data['republican_terms'], 1):
                print(f"  {i:2d}. {term_data['term']:<20} (coeff: {term_data['coefficient']:+.3f})")
            
            print(f"\nTOP DEMOCRAT-DISTINGUISHING TERMS:")
            print("  (Negative coefficients - algorithm associates these with Democrats)")
            for i, term_data in enumerate(data['democrat_terms'], 1):
                print(f"  {i:2d}. {term_data['term']:<20} (coeff: {term_data['coefficient']:+.3f})")
    
    def compare_words_across_years(self, start_congress, end_congress):
        """Compare top 15 words between two congress ranges and show frequency"""
        if not self.polarization_analysis:
            print("Run analyze_partisan_distinctions() first!")
            return
        
        # Filter congresses within the range
        congress_range = [c for c in self.polarization_analysis.keys() 
                         if start_congress <= c <= end_congress]
        
        if not congress_range:
            print(f"No congress data found between {start_congress} and {end_congress}")
            return
        
        # Collect all Republican and Democrat terms in the range
        republican_words = []
        democrat_words = []
        
        for congress_num in congress_range:
            data = self.polarization_analysis[congress_num]
            
            # Get top 15 terms for each party
            rep_terms = [t['term'] for t in data['republican_terms'][:15]]
            dem_terms = [t['term'] for t in data['democrat_terms'][:15]]
            
            republican_words.extend(rep_terms)
            democrat_words.extend(dem_terms)
        
        # Count frequencies
        rep_word_counts = Counter(republican_words)
        dem_word_counts = Counter(democrat_words)
        
        print(f"\nWORD FREQUENCY ANALYSIS: CONGRESS {start_congress} to {end_congress}")
        print("="*70)
        print(f"Analyzed {len(congress_range)} congress sessions: {sorted(congress_range)}")
        
        print(f"\nREPUBLICAN-DISTINGUISHING WORDS (appeared multiple times):")
        print("  Word                 Frequency")
        print("  " + "-"*35)
        rep_frequent = {word: count for word, count in rep_word_counts.items() if count > 1}
        if rep_frequent:
            for word, count in sorted(rep_frequent.items(), key=lambda x: x[1], reverse=True):
                print(f"  {word:<20} {count}")
        else:
            print("  No words appeared multiple times")
        
        print(f"\nDEMOCRAT-DISTINGUISHING WORDS (appeared multiple times):")
        print("  Word                 Frequency")
        print("  " + "-"*35)
        dem_frequent = {word: count for word, count in dem_word_counts.items() if count > 1}
        if dem_frequent:
            for word, count in sorted(dem_frequent.items(), key=lambda x: x[1], reverse=True):
                print(f"  {word:<20} {count}")
        else:
            print("  No words appeared multiple times")
        
        # Summary statistics
        print(f"\nSUMMARY:")
        print(f"  Total unique Republican words: {len(rep_word_counts)}")
        print(f"  Republican words appearing >1 time: {len(rep_frequent)}")
        print(f"  Total unique Democrat words: {len(dem_word_counts)}")
        print(f"  Democrat words appearing >1 time: {len(dem_frequent)}")
        
        return {
            'republican_frequencies': dict(rep_word_counts),
            'democrat_frequencies': dict(dem_word_counts),
            'congress_range': congress_range
        }
    
    def run_polarization_pipeline(self, top_n=15):
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
        
        return self.polarization_analysis

# Usage example
def main():
    pipeline = PolarizationAnalysisPipeline("feature_importance/congress_feature_importance_bigram_100_min_df_svm06-06.pkl")
    
    # Run full analysis
    analysis = pipeline.run_polarization_pipeline(top_n=20)
    
    # Example usage of word comparison function
    # Compare words between congress 100 and 110
    pipeline.compare_words_across_years(100, 110)
    
    return pipeline, analysis

if __name__ == "__main__":
    pipeline, analysis = main()