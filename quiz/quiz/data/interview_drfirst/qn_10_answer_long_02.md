# Fine-tuning BioBERT for Healthcare Text Classification


## 🔍 What is BioBERT?

BioBERT is a domain-specific language model pre-trained on biomedical literature (PubMed abstracts and PMC full-text articles) on top of the original BERT architecture. Unlike general BERT which was trained on Wikipedia and Books, BioBERT has specialized knowledge of biomedical terminology, relationships, and context.

## 🌟 Why BioBERT Over BERT for Healthcare?

1. **Domain-specific vocabulary**: BioBERT understands medical terminology like "myocardial infarction" or "adverse drug reaction" in proper context
2. **Biomedical relationships**: Better grasp of disease-symptom, drug-disease, and drug-drug interactions
3. **Performance**: Consistently outperforms general BERT on biomedical tasks with 10-15% improvement on benchmarks like BC5CDR and NCBI-disease

## 🛠️ The Fine-tuning Process for BioBERT

### 1. Data Preparation

- Collect labeled healthcare data relevant to your classification task:
  ```
  "My blood glucose has been elevated since starting prednisone" → Medication Side Effect
  "I need my metformin prescription renewed" → Medication Request
  "The incision site appears red and warm to touch" → Clinical Finding
  ```
- Clean and preprocess text (standardize abbreviations, normalize values)
- Consider healthcare-specific preprocessing: medication name normalization, PHI anonymization

### 2. Model Setup

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load BioBERT instead of general BERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-v1.1", 
    num_labels=num_classes
)
```

### 3. Tokenization

BioBERT's tokenizer has been adapted for biomedical text, with better subword tokenization for medical terms:

```python
# BioBERT tokenization handles medical terms better
max_length = 128  # Adjust based on your text length distribution
encoded_data = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)
```

### 4. Fine-tuning Configuration

BioBERT often requires different hyperparameters than general BERT:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./biobert_medical_classifier",
    num_train_epochs=4,             # Often needs more epochs than general BERT
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,               # Longer warmup for stability with medical terms
    weight_decay=0.01,              # Stronger regularization for specialized domain
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"      # F1 often better than accuracy for imbalanced medical classes
)
```

### 5. Class Weighting for Imbalanced Medical Data

Medical datasets often have severe class imbalance:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calculate class weights if your medical data is imbalanced (common)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(encoded_labels),
    y=encoded_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Use in loss function
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
```

### 6. Training Loop with Domain-specific Considerations

```python
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup

# Create dataset
dataset = TensorDataset(
    encoded_data["input_ids"], 
    encoded_data["attention_mask"], 
    torch.tensor(encoded_labels)
)

# Split data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Data loaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64)

# Optimizer with BioBERT-specific learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Create scheduler with longer warmup for stability
total_steps = len(train_dataloader) * 4  # epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,  # More warmup steps than regular BERT
    num_training_steps=total_steps
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(4):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

        model.zero_grad()
        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask,
            labels=b_labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Important for stability

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Average training loss: {avg_train_loss:.4f}")

    # Evaluate after each epoch
    model.eval()
    # Evaluation code here
```

### 7. Healthcare-specific Evaluation Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Run evaluation
model.eval()
predictions = []
true_labels = []

for batch in val_dataloader:
    b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

    with torch.no_grad():
        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_mask
        )

    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    true_labels.extend(b_labels.cpu().numpy())

# Generate comprehensive medical classification report
print(classification_report(true_labels, predictions, target_names=list(label2id.keys())))

# Confusion matrix - crucial for medical applications
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))

# For critical classes (e.g., urgent conditions), check sensitivity/specificity
for cls in range(num_classes):
    # Convert to binary problem for this class
    y_true_binary = [1 if l == cls else 0 for l in true_labels]
    y_pred_binary = [1 if p == cls else 0 for p in predictions]

    # Calculate metrics for this class
    tp = sum([1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1])
    tn = sum([1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0])
    fp = sum([1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1])
    fn = sum([1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0])

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Class: {id2label[cls]}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

# For imbalanced medical data, look at ROC AUC
# Convert predictions to one-hot
pred_one_hot = np.zeros((len(predictions), num_classes))
for i, pred in enumerate(predictions):
    pred_one_hot[i, pred] = 1

# Convert true labels to one-hot
true_one_hot = np.zeros((len(true_labels), num_classes))
for i, label in enumerate(true_labels):
    true_one_hot[i, label] = 1

# Calculate ROC AUC
roc_auc = roc_auc_score(true_one_hot, pred_one_hot, average='macro')
print(f"ROC AUC: {roc_auc:.4f}")
```

### 8. Inference with Confidence Thresholding

For medical applications, confidence thresholds are crucial:

```python
def predict_with_confidence(text, min_confidence=0.75):
    """For healthcare applications, we want high-confidence predictions"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

        if confidence.item() >= min_confidence:
            return id2label[prediction.item()], confidence.item()
        else:
            return "Uncertain classification", confidence.item()

# Example usage
clinical_text = "Patient reports intermittent chest pain radiating to left arm"
prediction, confidence = predict_with_confidence(clinical_text)
print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
```

### 9. Saving and Loading the Fine-tuned BioBERT Model

```python
# Save the model
model.save_pretrained("./biobert_healthcare_classifier")
tokenizer.save_pretrained("./biobert_healthcare_classifier")

# To load later
model = AutoModelForSequenceClassification.from_pretrained("./biobert_healthcare_classifier")
tokenizer = AutoTokenizer.from_pretrained("./biobert_healthcare_classifier")
```

## 🧠 Medical Use Case Advantages of BioBERT

1. **Better recognition of medical entities**: BioBERT correctly identifies "dyspnea" as a symptom rather than a disease
2. **Understanding medical abbreviations**: Recognizes "MI" as "myocardial infarction" not "Michigan"
3. **Contextual word representations**: Understands "discharge" differently in "hospital discharge" vs "wound discharge"
4. **Complex negation handling**: Better at understanding "Patient denies chest pain but reports dyspnea"

BioBERT's specialized medical knowledge allows it to achieve significantly higher accuracy on healthcare text classification tasks compared to general BERT, making it the preferred choice for clinical NLP applications.
