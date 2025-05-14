import re 
import os 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# You might already have some of these in pipeline_utils. Let's put core data loading/tokenization here.

class CongressSpeechDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, party_map={'D': 0, 'R': 1}, max_token_len: int = 512):
        
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


