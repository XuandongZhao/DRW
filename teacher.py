import argparse
import time
from tqdm import tqdm
import torch
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from model import BertClassifier, BertTokenClassifier
from data import load_imdb_data, load_token_data, load_glue_data
from wm import train_teacher, evaluate
from utils import epoch_time, POS_LIST, NER_LIST

parser = argparse.ArgumentParser("argument for training")
parser.add_argument("--task", type=str, choices=['pos', 'ner', 'imdb', 'sst2', 'mrpc', 'qnli'], default='sst2',
                    help="which task to run")
parser.add_argument('--seed', type=int, default=22, help='random seed')
parser.add_argument("--epochs", type=int, default=30, help="number of training epochs")
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--device", type=str, default='cuda:0', help="cuda device")
parser.add_argument("--batch-size", type=int, default=32, help="batch size for training")
parser.add_argument("--output-dir", type=str, default='./output/', help="path to save model")
parser.add_argument("--data-dir", type=str, default='./data/', help="path to training data")
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
print(f"Seed: {args.seed}")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

if args.task == 'imdb':
    train_dataloader, valid_dataloader, test_dataloader = load_imdb_data(args.data_dir, tokenizer, args.batch_size)
elif args.task == 'pos' or args.task == 'ner':
    train_dataloader, valid_dataloader, test_dataloader = load_token_data(tokenizer, args.task, args.batch_size)
elif args.task in GLUE_TASKS:
    train_dataloader, valid_dataloader, test_dataloader = load_glue_data(tokenizer, args.task, args.batch_size, start_i=0.0, end_i=1.0)
else:
    raise ValueError("Task is not defined. Must be one of pos, ner, imdb, glue")

bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device(args.device)

if args.task == 'imdb':
    model = BertClassifier(bert_model, num_class=2).to(device)
elif args.task == 'pos':
    model = BertTokenClassifier(bert_model, num_class=47).to(device)
elif args.task == 'ner':
    model = BertTokenClassifier(bert_model, num_class=9).to(device)
elif args.task in GLUE_TASKS:
    num_labels = 3 if args.task.startswith("mnli") else 1 if args.task == "stsb" else 2
    model = BertClassifier(bert_model, num_class=num_labels).to(device)
else:
    raise ValueError("Task is not defined. Must be one of pos, ner, imdb, glue")

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.2 * len(train_dataloader)),
                                            num_training_steps=len(train_dataloader) * args.epochs)
best_valid_acc = 0

for epoch in range(args.epochs):
    start_time = time.time()
    train_loss, train_pred, train_labels = train_teacher(model, train_dataloader, optimizer, scheduler, criterion, args.task, device)
    if args.task == 'pos' or args.task == 'ner':
        valid_loss, valid_pred, valid_labels = evaluate(model, test_dataloader, criterion, args.task, device)
    else:
        valid_loss, valid_pred, valid_labels = evaluate(model, valid_dataloader, criterion, args.task, device)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if args.task == 'imdb':
        metric = load_metric("accuracy")
        train_results = metric.compute(references=train_labels, predictions=train_pred)
        valid_results = metric.compute(references=valid_labels, predictions=valid_pred)
        train_acc = train_results['accuracy']
        valid_acc = valid_results['accuracy']
    elif args.task == 'pos' or args.task == 'ner':
        if args.task == 'pos':
            label_list = POS_LIST
        else:
            label_list = NER_LIST
        metric = load_metric("seqeval")
        train_pred_list = [[label_list[p] for (p, l) in zip(train_pred, train_labels) if l != -100]]
        train_label_list = [[label_list[l] for (p, l) in zip(train_pred, train_labels) if l != -100]]
        valid_pred_list = [[label_list[p] for (p, l) in zip(valid_pred, valid_labels) if l != -100]]
        valid_label_list = [[label_list[l] for (p, l) in zip(valid_pred, valid_labels) if l != -100]]
        train_results = metric.compute(predictions=train_pred_list, references=train_label_list)
        valid_results = metric.compute(predictions=valid_pred_list, references=valid_label_list)
        train_acc = train_results["overall_f1"]
        valid_acc = valid_results["overall_f1"]
    elif args.task in GLUE_TASKS:
        actual_task = "mnli" if args.task == "mnli-mm" else args.task
        metric = load_metric('glue', actual_task)
        train_results = metric.compute(references=train_labels, predictions=train_pred)
        valid_results = metric.compute(references=valid_labels, predictions=valid_pred)
        train_acc = train_results['accuracy']
        valid_acc = valid_results['accuracy']
    else:
        raise ValueError("Task is not defined. Must be one of pos, ner, imdb")

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        print("Get BEST result! Save model")
        torch.save(model.state_dict(), args.output_dir + args.task + '_teacher.pt')

print('BERT result of', args.task, best_valid_acc)
