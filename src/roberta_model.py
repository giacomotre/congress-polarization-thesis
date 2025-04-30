import torch
import torch.nn as nn
from transformers import AutoModel

class RobertaClassifier(nn.Module):
    def __init__(self, model_name: str = 'roberta-base', num_labels: int = 2):
        """
        Args:
            model_name (str): The name of the pre-trained RoBERTa model to use.
                            Defaults to 'roberta-base'.
            num_labels (int): The number of output classes (2 for Democrat/Republican).
        """
        super(RobertaClassifier, self).__init__()

        # Load the pre-trained RoBERTa model
        # We use AutoModel which loads the base transformer model (without classification head)
        self.roberta = AutoModel.from_pretrained(model_name)

        # Add a linear layer for classification -> but then used inside the def forward
        # The output size of roberta-base is 768 (hidden size)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

        # We might also add dropout for regularization, often applied before the classifier
        # self.dropout = nn.Dropout(0.1) # Example dropout with p=0.1

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs.
                                    Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Tensor containing attention mask.
                                        Shape: (batch_size, sequence_length)

        Returns:
            torch.Tensor: Logits for the classification classes.
                        Shape: (batch_size, num_labels)
        """
        # Pass inputs through the RoBERTa model
        # The output is a tuple, the first element is the sequence of hidden states
        # for each token, the second element is the pooled output ([CLS] token representation)
        # RoBERTa's pooled output (pooler_output) is often just the [CLS] token's hidden state
        # We'll use the [CLS] token's output for classification, which is the first token (index 0)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Get the hidden state of the [CLS] token (the first token)
        # outputs.last_hidden_state has shape (batch_size, sequence_length, hidden_size)
        cls_token_hidden_state = outputs.last_hidden_state[:, 0, :]

        # Optionally apply dropout
        # cls_token_hidden_state = self.dropout(cls_token_hidden_state)

        # Pass the [CLS] token's hidden state through the classifier layer
        logits = self.classifier(cls_token_hidden_state)

        return logits

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Instantiate the model
    # Ensure you have the transformers library installed: pip install transformers torch
    try:
        model = RobertaClassifier('roberta-base', num_labels=2)
        print("RoBERTa classifier model loaded successfully!")
        # print(model) # Uncomment to see the model architecture

        # Simulate a batch of tokenized inputs (e.g., batch size 2, sequence length 128)
        # In a real scenario, these would come from your DataLoader
        dummy_input_ids = torch.randint(0, model.roberta.config.vocab_size, (2, 128)) # Random token IDs
        dummy_attention_mask = torch.ones((2, 128)) # Assume all tokens are real (no padding) for simplicity in this dummy example
                                                    # Your DataLoader will provide correct attention masks

        # Move model and data to a device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dummy_input_ids = dummy_input_ids.to(device)
        dummy_attention_mask = dummy_attention_mask.to(device)

        # Perform a forward pass
        with torch.no_grad(): # No need to calculate gradients in example usage
            dummy_logits = model(dummy_input_ids, dummy_attention_mask)

        print("\nExample forward pass output (logits shape):", dummy_logits.shape)
        print("Example logits:\n", dummy_logits)

    except ImportError:
        print("Error: Please install the transformers library (`pip install transformers torch`).")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")