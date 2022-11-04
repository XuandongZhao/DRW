import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class IMDB_Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, new_labels=None):
        if new_labels is None:
            self.labels = [label for label in df['label']]
        else:
            self.labels = new_labels
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for
                      text in df['text']]
        self.embs = np.array(df['embs'].tolist())

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_embs(self, idx):
        return torch.tensor(self.embs[idx]).float()

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_embs = self.get_batch_embs(idx)

        return {'ids': batch_texts, 'labels': batch_y, 'embs': batch_embs}


class TOKEN_Dataset(torch.utils.data.Dataset):
    """
    The POSTaggingDataset object defined below is a PyTorch Dataset object for this task.
    Each example in the dataset is a feature dictionary, consisting of word piece ids, and corresponding label ids (labels).
    We associate a word's label with the last subword. Any remaining subwords, as well as special tokens like the start token or padding token, will have a label of -100 assigned to them.
    This will signal that we shouldn't compute a loss for that label.
    We also define a collate function that takes care of padding when examples are batched together.
    """

    def __init__(self, pos_data):
        self.sents = pos_data['input_ids']
        self.labels = pos_data['labels']

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        ids = torch.tensor(self.sents[index])
        labels = torch.tensor(self.labels[index])
        return {'ids': ids, 'labels': labels}

    @staticmethod
    def collate(batch, PAD_ID):
        ids = pad_sequence([item['ids'] for item in batch], batch_first=True, padding_value=PAD_ID)
        labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
        return {'ids': ids, 'labels': labels}


class GLUE_Dataset(torch.utils.data.Dataset):
    def __init__(self, glue_data, PAD_ID):
        self.sents = pad_sequence([torch.tensor(x) for x in glue_data['input_ids']], batch_first=True, padding_value=PAD_ID)
        self.atts = pad_sequence([torch.tensor(x) for x in glue_data['attention_mask']], batch_first=True, padding_value=0)
        self.labels = glue_data['label']

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        ids = self.sents[index]
        labels = self.labels[index]
        atts = self.atts[index]
        return {'ids': ids, 'labels': labels, 'atts': atts}


class Emo_Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, labels, new_labels=None):
        if new_labels is None:
            self.labels = [labels[label] for label in df['Emotion']]
        else:
            self.labels = new_labels
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for
                      text in df['Text']]
        self.embs = np.array(df['embs'].tolist())

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_embs(self, idx):
        return torch.tensor(self.embs[idx]).float()

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_embs = self.get_batch_embs(idx)

        return batch_texts, batch_y, batch_embs
