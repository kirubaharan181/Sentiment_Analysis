!pip install torch torchvision numpy pandas transformers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r"F:\SEM V\TCPC\Dataset\extract TV data in Section IV-A\allTV_review_2010.csv")
df = df[['Text', 'Star']]  # We need only text and stars for this task
df = df.head(1000)

def label_map(star):
    if star <= 2:
        return 0  # Negative
    elif star == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df['label'] = df['Star'].apply(label_map)
df[['Text', 'Star', 'label']].head()
# Cell 3: Define Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['Text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SentimentDataset(df, tokenizer)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
bert_model.to(device)
optimizer = optim.AdamW(bert_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()


epochs = 2
for epoch in range(epochs):
    bert_model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"BERT Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv2d(1, 100, (3, embed_dim))
        self.conv2 = nn.Conv2d(1, 100, (4, embed_dim))
        self.conv3 = nn.Conv2d(1, 100, (5, embed_dim))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids).unsqueeze(1)  # Add channel dimension
        x1 = nn.functional.relu(self.conv1(x)).squeeze(3)
        x1 = nn.functional.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = nn.functional.relu(self.conv2(x)).squeeze(3)
        x2 = nn.functional.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = nn.functional.relu(self.conv3(x)).squeeze(3)
        x3 = nn.functional.max_pool1d(x3, x3.size(2)).squeeze(2)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

text_cnn_model = TextCNN(vocab_size=tokenizer.vocab_size, embed_dim=128, num_classes=3).to(device)
optimizer = optim.Adam(text_cnn_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    text_cnn_model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = text_cnn_model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"TextCNN Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Use last hidden state
        logits = self.fc(x)
        return logits


textrnn_model = TextRNN(vocab_size=tokenizer.vocab_size, embed_dim=128, hidden_dim=64, num_classes=3).to(device)
optimizer = optim.Adam(textrnn_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    textrnn_model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = textrnn_model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"TextRNN Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:

            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            if 'attention_mask' in model.forward.__code__.co_varnames:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids)
            
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(true_labels, predictions), classification_report(true_labels, predictions)

distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
distilbert_model.to(device)

distilbert_dataset = SentimentDataset(df, distilbert_tokenizer)
distilbert_train_loader = DataLoader(distilbert_dataset, batch_size=16, shuffle=True)
distilbert_test_loader = DataLoader(distilbert_dataset, batch_size=16, shuffle=False)

for epoch in range(epochs):
    distilbert_model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in distilbert_train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = distilbert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"DistilBERT Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(distilbert_train_loader)}")



roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
roberta_model.to(device)

roberta_dataset = SentimentDataset(df, roberta_tokenizer)
roberta_train_loader = DataLoader(roberta_dataset, batch_size=16, shuffle=True)
roberta_test_loader = DataLoader(roberta_dataset, batch_size=16, shuffle=False)


for epoch in range(epochs):
    roberta_model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in roberta_train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"RoBERTa Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(roberta_train_loader)}")


print("Evaluating BERT model:")
bert_accuracy, bert_report = evaluate_model(bert_model, test_loader)
print("Accuracy:", bert_accuracy)
print(bert_report)

print("Evaluating TextCNN model:")
textcnn_accuracy, textcnn_report = evaluate_model(text_cnn_model, test_loader)
print("Accuracy:", textcnn_accuracy)
print(textcnn_report)

print("Evaluating TextRNN model:")
textrnn_accuracy, textrnn_report = evaluate_model(textrnn_model, test_loader)
print("Accuracy:", textrnn_accuracy)
print(textrnn_report)

print("Evaluating DistilBERT model:")
distilbert_accuracy, distilbert_report = evaluate_model(distilbert_model, distilbert_test_loader)
print("Accuracy:", distilbert_accuracy)
print(distilbert_report)

print("Evaluating RoBERTa model:")
roberta_accuracy, roberta_report = evaluate_model(roberta_model, roberta_test_loader)
print("Accuracy:", roberta_accuracy)
print(roberta_report)

