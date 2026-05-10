import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import json
import os
import time
import collections
from tqdm import tqdm  # Added for progress bar

# ======================
# CONFIGURATION
# ======================
# --- SAFETY LIMITS ---
MAX_FILE_SIZE_MB = 200  # Will strictly load only this much to save your 32GB RAM
MAX_VOCAB_SIZE = 20000  # Caps unique words to save your 8GB VRAM

# --- NEURAL NET CONFIG ---
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256
CONTEXT_LENGTH = 8  # Words to look back at

# --- TRAINING CONFIG ---
LEARNING_RATE = 0.001
BATCH_SIZE = 512  # High batch size to fully utilize your RTX 4060
EPOCHS = 3  # Kept low so it finishes in a reasonable time
GEN_TEMPERATURE = 0.7

# File paths
DATA_FILE = "dataset.txt"
MODEL_PATH = "model.pth"
VOCAB_PATH = "vocab.json"

# Auto-detect hardware accelerator
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ======================
# WORD-LEVEL TOKENIZER
# ======================
class WordTokenizer:
    def __init__(self):
        self.vocab = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.unk_idx = 0
        self.pad_idx = 1

    def build_vocab(self, text, max_vocab_size):
        words = re.findall(r"[\w']+|[.,!?;]", text.lower())

        # Count frequencies to keep only the most common words
        word_counts = collections.Counter(words)
        unique_words = [word for word, count in word_counts.most_common(max_vocab_size)]

        self.vocab = ['<unk>', '<pad>'] + unique_words
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {str(i): word for i, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        words = re.findall(r"[\w']+|[.,!?;]", text.lower())
        return [self.word_to_idx.get(word, self.unk_idx) for word in words]

    def decode(self, indices):
        words = [self.idx_to_word.get(str(i), '<unk>') for i in indices]
        if words:
            words[0] = words[0].capitalize()
            text = ''
            for word in words:
                if word in ['.', ',', '!', '?', ';']:
                    text = text.rstrip() + word + ' '
                else:
                    text += word + ' '
            return text.strip()
        return ''

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word
            }, f)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.vocab_size = len(self.vocab)


# ======================
# DATASET GENERATOR
# ======================
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, context_length):
        self.encoded = tokenizer.encode(text)
        self.context_length = context_length

    def __len__(self):
        return len(self.encoded) - self.context_length

    def __getitem__(self, idx):
        x = self.encoded[idx: idx + self.context_length]
        y = self.encoded[idx + self.context_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ======================
# NEURAL NETWORK
# ======================
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, context_length):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size * context_length, hidden_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# ======================
# GENERATION LOGIC
# ======================
def generate_response(model, tokenizer, prompt, max_length=20, temperature=GEN_TEMPERATURE):
    model.eval()
    encoded_prompt = tokenizer.encode(prompt)
    context = encoded_prompt[-CONTEXT_LENGTH:]

    # Pad if prompt is shorter than context length
    context = [tokenizer.pad_idx] * (CONTEXT_LENGTH - len(context)) + context
    response_indices = []

    with torch.no_grad():
        for _ in range(max_length):
            x_tensor = torch.tensor([context], dtype=torch.long).to(DEVICE)
            logits = model(x_tensor)[0]

            if temperature == 0:
                next_idx = torch.argmax(logits).item()
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=0)
                next_idx = torch.multinomial(probs, num_samples=1).item()

            response_indices.append(next_idx)
            context = context[1:] + [next_idx]

            next_word = tokenizer.idx_to_word.get(str(next_idx), '')
            if next_word in ['.', '?', '!']:
                break

    return tokenizer.decode(response_indices)


# ======================
# MAIN EXECUTION
# ======================
def main():
    print(f"Using device: {DEVICE}")
    tokenizer = WordTokenizer()

    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        print("\nFound existing model. Loading weights and vocabulary...")
        tokenizer.load(VOCAB_PATH)
        model = SimpleLLM(tokenizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, CONTEXT_LENGTH)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        print("Model loaded successfully!")

    else:
        print("\nNo existing model found. Beginning training phase.")

        if not os.path.exists(DATA_FILE):
            print(f"Error: Could not find {DATA_FILE}.")
            return

        # --- SAFE FILE LOADING LOGIC ---
        file_size_bytes = os.path.getsize(DATA_FILE)
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

        bytes_to_read = min(file_size_bytes, max_bytes)
        mb_to_read = bytes_to_read / (1024 * 1024)
        total_mb = file_size_bytes / (1024 * 1024)

        print(f"\nFile size detected: {total_mb:.2f} MB")
        if file_size_bytes > max_bytes:
            print(
                f"WARNING: File is too large! Safely loading only the first {mb_to_read:.2f} MB to prevent RAM crash.")
        else:
            print(f"Loading {mb_to_read:.2f} MB of data...")

        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            text = f.read(bytes_to_read)

        print("Data loaded. Building vocabulary...")
        tokenizer.build_vocab(text, MAX_VOCAB_SIZE)
        tokenizer.save(VOCAB_PATH)
        print(f"Vocabulary built and capped at: {tokenizer.vocab_size} words.")

        dataset = TextDataset(text, tokenizer, CONTEXT_LENGTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = SimpleLLM(tokenizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, CONTEXT_LENGTH).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        model.train()
        start_time = time.time()

        for epoch in range(EPOCHS):
            total_loss = 0

            # Progress bar for the terminal
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)

            for batch_x, batch_y in progress_bar:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{EPOCHS}] Completed - Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"\nTraining complete in {time.time() - start_time:.2f} seconds.")
        print(f"Model saved to {MODEL_PATH}")

    # --- INTERACTIVE CHAT ---
    print("\n" + "=" * 60)
    print("AI READY - INTERACTIVE MODE")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            if not user_input:
                continue

            print("AI:", end=" ", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()