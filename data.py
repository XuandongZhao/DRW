import torch
import pandas as pd
from datasets import load_dataset
from dataset import IMDB_Dataset, TOKEN_Dataset, GLUE_Dataset


def load_imdb_data(data_path, tokenizer, batch_size=8, shuffle=True, start_i=0.0, end_i=1.0):
    df_train = pd.read_pickle(data_path + 'imdb_train.pkl')
    df_valid = pd.read_pickle(data_path + 'imdb_valid.pkl')
    df_test = pd.read_pickle(data_path + 'imdb_test.pkl')
    len_train = len(df_train)
    len_valid = len(df_valid)
    len_test = len(df_test)
    train_data = IMDB_Dataset(df_train[int(start_i * len_train):int(end_i * len_train)], tokenizer)
    valid_data = IMDB_Dataset(df_valid[int(start_i * len_valid):int(end_i * len_valid)], tokenizer)
    test_data = IMDB_Dataset(df_test[int(start_i * len_test):int(end_i * len_test)], tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    print('Finish loading IMDB dataset! Train, valid, and test data length are:')
    print(len(train_data), len(valid_data), len(test_data))
    return train_dataloader, valid_dataloader, test_dataloader


def load_token_data(tokenizer, task, batch_size=8, shuffle=True, start_i=0.0, end_i=1.0):
    datasets = load_dataset("conll2003", revision="master")

    def tokenize_and_align_labels(examples, label_all_tokens=True):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
    PAD_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    len_train = len(tokenized_datasets['train'])
    len_valid = len(tokenized_datasets['validation'])
    len_test = len(tokenized_datasets['test'])

    train_dataset = TOKEN_Dataset(tokenized_datasets['train'][int(start_i * len_train):int(end_i * len_train)])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                                   collate_fn=lambda batch: train_dataset.collate(batch, PAD_ID))
    valid_dataset = TOKEN_Dataset(tokenized_datasets['validation'][int(start_i * len_valid):int(end_i * len_valid)])
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle,
                                                   collate_fn=lambda batch: valid_dataset.collate(batch, PAD_ID))
    test_dataset = TOKEN_Dataset(tokenized_datasets['test'][int(start_i * len_test):int(end_i * len_test)])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                                  collate_fn=lambda batch: test_dataset.collate(batch, PAD_ID))
    print(f'Finish loading {task} dataset! Train, valid, and test data length are:')
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    return train_dataloader, valid_dataloader, test_dataloader


def load_glue_data(tokenizer, task, batch_size=8, shuffle=True, start_i=0.0, end_i=1.0):
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[task]
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    PAD_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    len_train = len(tokenized_datasets['train'])
    len_valid = len(tokenized_datasets['validation'])
    len_test = len(tokenized_datasets['test'])

    train_dataset = GLUE_Dataset(tokenized_datasets['train'][int(start_i * len_train):int(end_i * len_train)], PAD_ID)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataset = GLUE_Dataset(tokenized_datasets['validation'][int(start_i * len_valid):int(end_i * len_valid)], PAD_ID)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataset = GLUE_Dataset(tokenized_datasets['test'][int(start_i * len_test):int(end_i * len_test)], PAD_ID)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    print(f'Finish loading {task} dataset! Train, valid, and test data length are:')
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    return train_dataloader, valid_dataloader, test_dataloader
