import argparse
import time
import torch
import numpy as np
import json
from datasets import load_metric
from transformers import AutoTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from model import BertClassifier, BertTokenClassifier
from data import load_imdb_data, load_token_data, load_glue_data
from wm import train_teacher, evaluate, train_student_soft, train_student_hard, train_student_hard_wm, train_student_soft_wm, teacher_hard_acc
from utils import epoch_time, KLLoss, NpEncoder, POS_LIST, NER_LIST

parser = argparse.ArgumentParser("argument for training")
parser.add_argument("--task", type=str, choices=['pos', 'ner', 'imdb', 'sst2', 'mrpc', 'qnli'], default='mrpc', help="which task to run")
parser.add_argument("--wm", type=int, default=1, help="whether to use watermark")
parser.add_argument('--wmidx', type=int, nargs='+', default=[0], help="which idx to embed")
parser.add_argument("--hard", type=int, default=1, help="distillation with hard labels")
parser.add_argument('--tseed', type=int, default=22, help='random seed for torch')
parser.add_argument('--nseed', type=int, default=15, help='random seed for numpy')
parser.add_argument("--starti", type=float, default=0.0, help="start ratio of the training data")
parser.add_argument("--endi", type=float, default=0.5, help="end ratio of the training data")
parser.add_argument("--sub", type=float, default=0.5, help="subset of the x")
parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--eps", type=float, default=0.2, help="epsilon in wm")
parser.add_argument("--k", type=float, default=16, help="frequency in wm")
parser.add_argument("--device", type=str, default='cuda:6', help="cuda device")
parser.add_argument("--batch-size", type=int, default=256, help="batch size for training")
parser.add_argument("--tid", type=int, default=2, help="token id for sentence")
parser.add_argument("--output-dir", type=str, default='./output/', help="path to save model")
parser.add_argument("--data-dir", type=str, default='./data/', help="path to training data")
args = parser.parse_args()
save_perfix = f"{args.output_dir}{args.task}_wm-{args.wm}_hard-{args.hard}_eps-{args.eps}_k-{args.k}_tseed-{args.tseed}_nseed-{args.nseed}_sub-{args.sub}_epochs-{args.epochs}"
print(args)
print(save_perfix + f"_wmidx-{args.wmidx}")

torch.manual_seed(args.tseed)
np.random.seed(args.nseed)

print(f"Seed for torch\t: {args.tseed}")
print(f"Seed for np\t: {args.nseed}")
print(f"Task\t: {args.task}")
print(f"WM\t: {args.wm}")
print(f"Hard\t: {args.hard}")
print(f"K\t: {args.k}")
print(f"eps\t: {args.eps}")
print(f"WMidx\t: {args.wmidx}")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
teacher_bert = BertModel.from_pretrained('bert-base-uncased')
student_bert = BertModel.from_pretrained('bert-base-uncased')
device = torch.device(args.device)
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

if args.task == 'imdb':
    train_dataloader, valid_dataloader, test_dataloader = load_imdb_data(args.data_dir, tokenizer, args.batch_size, start_i=args.starti, end_i=args.endi)
    cur_num_class = 2
    teacher_model = BertClassifier(teacher_bert, num_class=cur_num_class).to(device)
    student_model = BertClassifier(student_bert, num_class=cur_num_class).to(device)
elif args.task == 'pos':
    train_dataloader, valid_dataloader, test_dataloader = load_token_data(tokenizer, args.task, args.batch_size, start_i=args.starti, end_i=args.endi)
    cur_num_class = 47
    teacher_model = BertTokenClassifier(teacher_bert, num_class=cur_num_class).to(device)
    student_model = BertTokenClassifier(student_bert, num_class=cur_num_class).to(device)
elif args.task == 'ner':
    train_dataloader, valid_dataloader, test_dataloader = load_token_data(tokenizer, args.task, args.batch_size, start_i=args.starti, end_i=args.endi)
    cur_num_class = 9
    teacher_model = BertTokenClassifier(teacher_bert, num_class=cur_num_class).to(device)
    student_model = BertTokenClassifier(student_bert, num_class=cur_num_class).to(device)
elif args.task in GLUE_TASKS:
    train_dataloader, valid_dataloader, test_dataloader = load_glue_data(tokenizer, args.task, args.batch_size, start_i=args.starti, end_i=args.endi)
    cur_num_class = 3 if args.task.startswith("mnli") else 1 if args.task == "stsb" else 2
    teacher_model = BertClassifier(teacher_bert, num_class=cur_num_class).to(device)
    student_model = BertClassifier(student_bert, num_class=cur_num_class).to(device)
else:
    raise ValueError("Task is not defined. Must be one of pos, ner, imdb")

teacher_model.load_state_dict(torch.load(f"{args.output_dir}{args.task}_teacher.pt", map_location=args.device))
print('Load teacher from', f"{args.output_dir}{args.task}_teacher.pt")
if args.hard:
    criterion = torch.nn.CrossEntropyLoss()
else:
    criterion = KLLoss(num_classes=cur_num_class)

# optimizer = AdamW(student_model.parameters(), lr=args.lr, eps=1e-8)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.2 * len(train_dataloader)),
#                                             num_training_steps=len(train_dataloader) * args.epochs)
optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
scheduler = None
best_valid_loss = float('inf')

key = np.random.rand(128)
vec = np.random.randn(30522, 128)
skey = np.load('select_key.npy')
log_history = []
for epoch in range(args.epochs):
    start_time = time.time()

    if args.wm:
        if args.hard:
            train_loss, train_pred, train_labels = train_student_hard_wm(student_model, teacher_model, vec, key, skey, train_dataloader, optimizer, scheduler, criterion, \
                                                                         args.task, cur_num_class, device, k=args.k, epsilon=args.eps, tid=args.tid, wmidx=args.wmidx, sub=args.sub)
        else:
            train_loss, train_pred, train_labels = train_student_soft_wm(student_model, teacher_model, vec, key, skey, train_dataloader, optimizer, scheduler, criterion, \
                                                                         args.task, cur_num_class, device, k=args.k, epsilon=args.eps, tid=args.tid, wmidx=args.wmidx, sub=args.sub)
    else:
        if args.hard:
            train_loss, train_pred, train_labels = train_student_hard(student_model, teacher_model, train_dataloader, optimizer, scheduler, criterion, \
                                                                      args.task, cur_num_class, device)
        else:
            train_loss, train_pred, train_labels = train_student_soft(student_model, teacher_model, train_dataloader, optimizer, scheduler, criterion, \
                                                                      args.task, cur_num_class, device)

    eval_criterion = torch.nn.CrossEntropyLoss()
    valid_loss, valid_pred, valid_labels = evaluate(student_model, valid_dataloader, eval_criterion, args.task, device)
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    torch.save(student_model.state_dict(), save_perfix + ".pt")
    print(save_perfix + f"_wmidx-{args.wmidx}")
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
    epoch_log = {"Training": f"Epoch:{epoch + 1:02} Train Loss:{train_loss:.4f} Train Acc:{train_acc:.4f} Valid Loss:{valid_loss:.4f} Valid Acc:{valid_acc:.4f}"}
    log_history.append(epoch_log)

torch.save(student_model.state_dict(), save_perfix + ".pt")

eval_criterion = torch.nn.CrossEntropyLoss()
if args.task == 'sst2' or args.task == 'mrpc':
    test_iterator = valid_dataloader
else:
    test_iterator = test_dataloader

student_loss, student_pred, student_labels = evaluate(student_model, test_iterator, eval_criterion, args.task, device)
teacher_loss, teacher_pred, teacher_labels = evaluate(teacher_model, test_iterator, eval_criterion, args.task, device)
if args.task == 'imdb':
    metric = load_metric("accuracy")
    student_result = metric.compute(references=student_labels, predictions=student_pred)
    teacher_result = metric.compute(references=teacher_labels, predictions=teacher_pred)
elif args.task == 'pos' or args.task == 'ner':
    if args.task == 'pos':
        label_list = POS_LIST
    else:
        label_list = NER_LIST
    metric = load_metric("seqeval")
    student_pred_list = [[label_list[p] for (p, l) in zip(student_pred, student_labels) if l != -100]]
    student_label_list = [[label_list[l] for (p, l) in zip(student_pred, student_labels) if l != -100]]
    teacher_pred_list = [[label_list[p] for (p, l) in zip(teacher_pred, teacher_labels) if l != -100]]
    teacher_label_list = [[label_list[l] for (p, l) in zip(teacher_pred, teacher_labels) if l != -100]]
    student_results = metric.compute(predictions=student_pred_list, references=student_label_list)
    teacher_results = metric.compute(predictions=teacher_pred_list, references=teacher_label_list)
    student_result = student_results["overall_f1"]
    teacher_result = teacher_results["overall_f1"]
elif args.task in GLUE_TASKS:
    actual_task = "mnli" if args.task == "mnli-mm" else args.task
    metric = load_metric('glue', actual_task)
    student_result = metric.compute(references=student_labels, predictions=student_pred)
    teacher_result = metric.compute(references=teacher_labels, predictions=teacher_pred)
else:
    raise ValueError("Task is not defined. Must be one of pos, ner, imdb")
print('Student', student_result)
print('Teacher', teacher_result)
test_results = {}
if args.hard:
    test_loss, test_pred, test_labels = teacher_hard_acc(teacher_model, vec, key, skey, test_iterator, args.task, cur_num_class, device,
                                                         k=args.k, epsilon=args.eps, tid=args.tid, wmidx=args.wmidx)
    if args.task == 'imdb':
        metric = load_metric("accuracy")
        test_results = metric.compute(references=test_labels, predictions=test_pred)
        test_acc = test_results['accuracy']
    elif args.task == 'pos' or args.task == 'ner':
        if args.task == 'pos':
            label_list = POS_LIST
        else:
            label_list = NER_LIST
        metric = load_metric("seqeval")
        test_pred_list = [[label_list[p] for (p, l) in zip(test_pred, test_labels) if l != -100]]
        test_label_list = [[label_list[l] for (p, l) in zip(test_pred, test_labels) if l != -100]]
        test_results = metric.compute(predictions=test_pred_list, references=test_label_list)
        test_acc = test_results["overall_f1"]
    elif args.task in GLUE_TASKS:
        actual_task = "mnli" if args.task == "mnli-mm" else args.task
        metric = load_metric('glue', actual_task)
        test_results = metric.compute(references=test_labels, predictions=test_pred)
    else:
        raise ValueError("Task is not defined. Must be one of pos, ner, imdb")
    print('Teacher hard results', test_results)

log_history.append({"Student test": student_result, "Teacher test": teacher_result, "Teacher hard": test_results})
json.dump(log_history, open(save_perfix + "_log_history.json", "w"), indent=4, ensure_ascii=False, cls=NpEncoder)
