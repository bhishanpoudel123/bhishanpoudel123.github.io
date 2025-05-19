# How to deal when document is longer than the contenxt window


### 🔁 **1. Sliding Window (Chunking with Overlap)**

This splits a long document into overlapping 512-token chunks.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Simulate long text
long_text = "This is a very long document " * 300  # > 512 tokens

# Tokenize with sliding window
tokens = tokenizer(
    long_text,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    max_length=512,
    stride=128,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

# Loop through all input chunks
for i in range(len(tokens['input_ids'])):
    input_ids = tokens['input_ids'][i].unsqueeze(0)
    attention_mask = tokens['attention_mask'][i].unsqueeze(0)

    # Pass to model
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    print(f"Chunk {i+1}: Predicted Class = {predicted_class}")
```



### ✂️ **2. Summarization Preprocessing (e.g., with T5)**

Use a pretrained summarizer to compress long text before classification.

```python
from transformers import pipeline

# Load a T5/BART summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

long_text = "Your very long medical note..." * 200

# Generate summary
summary = summarizer(long_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

# Use summary as input to classifier
tokens = tokenizer(summary, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    output = model(**tokens)
    predicted_class = torch.argmax(output.logits, dim=1).item()

print(f"Predicted Class from Summary: {predicted_class}")
```



### 🧠 **3. Use Longformer for Long Sequences**

Switch to Longformer to process longer texts natively (e.g., 4096 tokens).

```python
from transformers import LongformerTokenizer, LongformerForSequenceClassification

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=4)

long_text = "Your really long document..." * 1000

# Tokenize for Longformer
inputs = tokenizer(long_text, return_tensors="pt", truncation=True, padding="max_length", max_length=4096)

# Forward pass
with torch.no_grad():
    output = model(**inputs)
    predicted_class = torch.argmax(output.logits, dim=1).item()

print(f"Predicted Class with Longformer: {predicted_class}")
```



Let me know which of these fits your pipeline best, or if you'd like a comparison of latency/accuracy trade-offs between them.



**Q: How do you deal with long documents that exceed the context window using sliding window/chunks?**

**Answer:**

When working with models like BERT that have a fixed input length (typically 512 tokens), long medical documents like clinical notes often exceed this limit. To handle such cases, I use a **sliding window approach**, also known as **chunking with overlap**.

This method involves breaking the long document into multiple overlapping chunks that fit within the model’s context window. Each chunk is passed through the model independently, and the results are aggregated.



### ✅ **Why Sliding Window?**

* It ensures **full document coverage** without losing information.
* Overlapping helps **preserve context** across chunk boundaries.
* It’s simple and model-agnostic—no architecture changes required.



### ⚙️ **Approach**

1. **Tokenize the full document.**
2. **Create overlapping chunks** (e.g., chunk of 512 tokens with 128-token stride).
3. Pass each chunk to the model.
4. **Aggregate predictions** — e.g., majority vote, max confidence, or weighted average.



### 🧪 **Python Code Template**

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
model.eval()

# Simulated long clinical document
long_text = "Patient reports dizziness and nausea after taking medication..." * 100  # too long

# Tokenize using sliding window
tokens = tokenizer(
    long_text,
    return_overflowing_tokens=True,
    stride=128,
    max_length=512,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

# Loop through chunks
all_logits = []
for i in range(len(tokens['input_ids'])):
    input_ids = tokens['input_ids'][i].unsqueeze(0)
    attention_mask = tokens['attention_mask'][i].unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        all_logits.append(logits)

# Aggregate (e.g., mean of logits)
avg_logits = torch.mean(torch.stack(all_logits), dim=0)
predicted_class = torch.argmax(avg_logits, dim=1).item()

print(f"Predicted class from sliding window: {predicted_class}")
```



### 🧠 **When Do I Use This?**

I use this method in medical NLP tasks like:

* Clinical note classification
* Triage message categorization
* Symptom and medication detection

Where losing part of the document could mean missing a key symptom or condition.



### 📌 **Benefits**

* Preserves all context
* Doesn't require changing model architecture
* Easily integrates with existing BERT-based pipelines
