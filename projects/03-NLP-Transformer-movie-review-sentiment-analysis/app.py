import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
import math
import os

# --- 1. Configuration (Should match training script) ---
# This part is crucial. The model architecture and parameters here
# must be IDENTICAL to the ones used for training the model.
VOCAB_SIZE = 10000
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
N_CLASSES = 2
N_HEADS = 4
N_ENCODER_LAYERS = 6
DROPOUT = 0.1


# --- 2. Model Architecture Definition ---
# We must redefine the exact same model structure so we can load our saved weights into it.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_encoder_layers, hidden_dim, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_encoder_layers)
        self.fc = nn.Linear(embed_dim, output_dim)
        self.d_model = embed_dim

    def forward(self, text, src_key_padding_mask):
        embedded = self.embedding(text) * math.sqrt(self.d_model)
        positioned = self.pos_encoder(embedded)
        encoder_output = self.transformer_encoder(positioned, src_key_padding_mask=src_key_padding_mask)
        pooled_output = encoder_output.mean(dim=1)
        return self.fc(pooled_output)


# --- 3. Loading Artifacts (The "Model Loading" Step) ---
# This section loads all the necessary components to make predictions.
# This happens ONCE when the API server starts up.

print("Loading model and artifacts...")

# Set device (inference is often faster on CPU for single requests)
DEVICE = torch.device("cpu")


# --- Build Vocabulary ---
# We need the exact same vocabulary as used in training to convert text to integers.
def build_vocabulary():
    print("Building vocabulary from IMDB dataset...")
    dataset = load_dataset("imdb", split="train")
    tokenizer = get_tokenizer('basic_english')

    def yield_tokens(data_iter):
        for item in data_iter:
            yield tokenizer(item['text'])

    vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>", "<pad>"], max_tokens=VOCAB_SIZE)
    vocab.set_default_index(vocab["<unk>"])
    print("Vocabulary built.")
    return vocab, tokenizer


vocab, tokenizer = build_vocabulary()
PAD_IDX = vocab['<pad>']
MODEL_SAVE_PATH = "sentiment_transformer.pth"

# --- Load Model ---
# Instantiate the model architecture
model = TransformerSentimentClassifier(
    len(vocab), EMBEDDING_DIM, N_HEADS, N_ENCODER_LAYERS, HIDDEN_DIM, N_CLASSES, DROPOUT, PAD_IDX
).to(DEVICE)

# Load the trained weights
if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded successfully from {MODEL_SAVE_PATH}")
else:
    print(f"Warning: Model file not found at {MODEL_SAVE_PATH}. The API will use an untrained model.")


# --- 4. API Definition (FastAPI) ---

# Define the input data model using Pydantic
class ReviewRequest(BaseModel):
    text: str


# Define the output data model
class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float


# Create the FastAPI application
app = FastAPI(title="Sentiment Analysis API", description="An API to predict the sentiment of a movie review.")


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    """
    Takes a movie review text and returns the predicted sentiment (Positive/Negative).
    """
    # Get the text from the request
    text = request.text

    # 1. Tokenize the text
    tokenized_text = tokenizer(text)

    # 2. Numericalize using the vocabulary
    indexed_text = [vocab[token] for token in tokenized_text]

    # 3. Convert to tensor and add batch dimension
    tensor = torch.LongTensor(indexed_text).unsqueeze(0).to(DEVICE)

    # 4. Create padding mask
    # For a single prediction, the mask is just all False (no padding to ignore)
    padding_mask = (tensor == PAD_IDX)  # This will be all False

    # 5. Make prediction
    with torch.no_grad():
        prediction = model(tensor, padding_mask)

        # 6. Get probabilities using Softmax
        probabilities = torch.softmax(prediction, dim=1)

        # 7. Get the predicted class and its confidence
        confidence, predicted_class = torch.max(probabilities, dim=1)

        sentiment = "Positive" if predicted_class.item() == 1 else "Negative"

    return {"sentiment": sentiment, "confidence": confidence.item()}


print("API is ready to accept requests.")
