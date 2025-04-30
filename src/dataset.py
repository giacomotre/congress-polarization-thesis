import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re # Import regex for basic text cleaning
import os # Import os for file path joining

# You might already have some of these in pipeline_utils. Let's put core data loading/tokenization here.

class CongressSpeechDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, party_map={'D': 0, 'R': 1}, max_token_len: int = 512):
        """
        Args:
            dataframe (pd.DataFrame): Pandas DataFrame containing speech data.
                                    Must have 'speech' and 'party' columns.
                                    Assumes initial filtering (min word count)
                                    and duplicate removal has been done prior.
            tokenizer: The RoBERTa tokenizer (e.g., AutoTokenizer.from_pretrained('roberta-base')).
            party_map (dict): Dictionary mapping party strings ('D', 'R') to numerical labels (0, 1).
            max_token_len (int): The maximum sequence length for tokenization. Speeches will be truncated.
        """
        if not all(col in dataframe.columns for col in ['speech', 'party']):
            raise ValueError("DataFrame must contain 'speech' and 'party' columns.")

        self.tokenizer = tokenizer
        self.party_map = party_map
        self.max_token_len = max_token_len

        # Ensure we only have the parties we intend to classify and map them
        valid_parties = list(self.party_map.keys())
        self.data = dataframe[dataframe['party'].isin(valid_parties)].copy()

        # Handle potential NaN values in speech column before processing
        self.data['speech'] = self.data['speech'].fillna('')

        # Simple cleaning adapted from your pipeline_utils for BERT compatibility (less aggressive)
        # We primarily remove excess whitespace here. RoBERTa handles punctuation/casing.
        self.data['speech'] = self.data['speech'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())


        self.data['label'] = self.data['party'].map(self.party_map)

        # Convert to list of dictionaries for easy iteration
        self.data_list = self.data[['speech', 'label']].to_dict('records')

        print(f"Initialized Dataset with {len(self.data_list)} samples.")


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieves a single sample, tokenizes it, and returns model inputs and label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]
        text = sample['speech']
        label = sample['label']

        # Tokenize the text
        # truncation=True handles texts longer than max_token_len
        # padding='max_length' pads shorter texts to max_token_len
        encoding = self.tokenizer(
            text,
            add_special_tokens=True, # Add [CLS] and [SEP] tokens
            max_length=self.max_token_len,
            padding='max_length',    # Pad to max_length
            truncation=True,         # Truncate if longer than max_length
            return_attention_mask=True,# Return attention mask, which tells the BERT model which tokens are real text and which are padding tokens
            return_tensors='pt'      # Return PyTorch tensors
        )

        # Squeeze removes the batch dimension added by return_tensors='pt'
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long) # Ensure label is a tensor
        }


# Example Usage (for testing purposes)
if __name__ == '__main__':
    # This part simulates how you would use the dataset in your main script

    # 1. Create a dummy DataFrame or load a small subset of your data
    #    In your actual pipeline, you would load a specific year's CSV
    #    and apply filtering/speaker split BEFORE creating the dataset.
    dummy_data = {
        'speech_id': [1, 2, 3, 4, 5, 6],
        'speech': [
            "This is a short democratic speech.",
            "Republicans believe in lower taxes.",
            "This speech is very long and exceeds the maximum token length of 512 for the roberta base model. It contains many sentences and repetitions to ensure it gets truncated. This allows us to test the truncation functionality of the dataset. We need to make sure that the tokenizer handles this case correctly, providing input IDs and attention masks of the expected size. Without proper truncation, long texts would cause errors during model forward pass. Padding ensures shorter texts also match the required input size.",
            "Another democratic speech.",
            "Short R speech.",
            "Too short", # This should be filtered out if min_word_count >= 3
        ],
        'speakerid': [101, 102, 103, 104, 105, 106],
        'lastname': ['Smith', 'Jones', 'Brown', 'Green', 'White', 'Black'],
        'firstname': ['John', 'Mary', 'Peter', 'Lisa', 'David', 'Sarah'],
        'chamber': ['H', 'H', 'H', 'H', 'H', 'H'],
        'party': ['D', 'R', 'D', 'D', 'R', 'D'],
        'year': [1990, 1990, 1990, 1990, 1990, 1990],
        'char_count': [30, 30, 1000, 30, 30, 8],
        'word_count': [6, 5, 180, 4, 4, 2] # Include word_count for filtering simulation
    }
    dummy_df = pd.DataFrame(dummy_data)

    # Simulate applying filtering BEFORE creating the Dataset
    # We can use the filter_short_speeches function from your pipeline_utils.py
    # Make sure pipeline_utils is in your Python path or copy the function here for the example
    def filter_short_speeches_example(df: pd.DataFrame, text_col: str = "speech", min_words: int = 15) -> pd.DataFrame:
        # Simplified version for example - doesn't return count
        return df[df[text_col].apply(lambda x: len(str(x).split()) >= min_words)].copy()

    min_words_filter = 15
    filtered_df = filter_short_speeches_example(dummy_df, min_words=min_words_filter)
    # Add duplicate filtering here too if needed before splitting/dataset creation

    # 2. Initialize the RoBERTa tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # 3. Create an instance of the dataset using the filtered DataFrame
    #    In your pipeline, you would pass the train/val/test dataframes obtained
    #    AFTER the leave-out-speaker split for a specific year.
    try:
        dataset = CongressSpeechDataset(filtered_df, tokenizer, max_token_len=128) # Use a smaller max_token_len for faster example
        print(f"Dataset created with {len(dataset)} samples.")

        # 4. Create a DataLoader (useful for iterating in batches)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 5. Iterate through a batch to see the output format
        print("\nExample batch from DataLoader:")
        for batch in dataloader:
            print("Input IDs shape:", batch['input_ids'].shape)
            print("Attention Mask shape:", batch['attention_mask'].shape)
            print("Labels shape:", batch['labels'].shape)
            print("Input IDs (first sample):", batch['input_ids'][0][:20], "...") # Print first 20 tokens
            print("Attention Mask (first sample):", batch['attention_mask'][0][:20], "...") # Print first 20 masks
            print("Labels:", batch['labels'])
            break # Just show one batch

    except ValueError as e:
        print(f"Could not create Dataset: {e}")