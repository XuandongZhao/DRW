from torch import nn


class BertClassifier(nn.Module):

    def __init__(self, model, num_class=6, hidden=768, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden, num_class)

    def forward(self, input_id, mask):
        embedded = self.dropout(self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)[1])
        predictions = self.linear(embedded)
        return predictions


class BertTokenClassifier(nn.Module):
    def __init__(self, bert, num_class, dropout=0.5):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.linear = nn.Linear(embedding_dim, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.dropout(self.bert(text)[0])
        # embedded = [batch size, sent len, emb dim]
        predictions = self.linear(self.dropout(embedded))
        # predictions = [batch size, sent len, output dim]
        return predictions


class BERTPoSTagger(nn.Module):
    def __init__(self, bert, output_dim, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        text = text.permute(1, 0)
        # text = [batch size, sent len]
        embedded = self.dropout(self.bert(text)[0])
        # embedded = [batch size, seq len, emb dim]
        embedded = embedded.permute(1, 0, 2)
        # embedded = [sent len, batch size, emb dim]
        predictions = self.fc(self.dropout(embedded))
        # predictions = [sent len, batch size, output dim]
        return predictions
