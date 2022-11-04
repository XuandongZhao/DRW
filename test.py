import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from model import BertClassifier, BertTokenClassifier
from data import load_imdb_data, load_token_data, load_glue_data
from wm import train_teacher, evaluate, train_student_soft, train_student_hard, softmax_signal_wm, teacher_hard_acc
from utils import epoch_time, KLLoss, build_periodogram, generate_plots_nb, get_spectrum_window, POS_LIST, NER_LIST
import glob
import json
import scipy


def test_detect(teacher_name, student_name, device):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    teacher_bert = BertModel.from_pretrained('bert-base-uncased')
    student_bert = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device(device)
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    task = teacher_name.split("/")[2].split("_")[0]
    batch_size = 2
    if 'wm-1' in student_name:
        tseed = int(student_name.split('tseed-')[1][:2])
        nseed = int(student_name.split('nseed-')[1][:2])
    else:
        tseed = 11222
        nseed = 11222

    if task == 'pos':
        target_class = 22
        wmidx = [22]
        ratio = 0.98
    elif task == 'ner':
        target_class = 2
        wmidx = [2]
        ratio = 0.98
    elif task == 'mrpc':
        target_class = 0
        wmidx = [0]
        ratio = 0.95
    elif task == 'sst2':
        target_class = 0
        wmidx = [0]
        ratio = 0.98
    tid = 2
    sub = 0.5
    k = 16
    epsilon = 0.2
    torch.manual_seed(tseed)
    np.random.seed(nseed)

    key = np.random.rand(128)
    vec = np.random.randn(30522, 128)
    skey = np.load('select_key.npy')

    if task == 'pos':
        cur_num_class = 47
        teacher_model = BertTokenClassifier(teacher_bert, num_class=cur_num_class).to(device)
        student_model = BertTokenClassifier(student_bert, num_class=cur_num_class).to(device)
    elif task == 'ner':
        cur_num_class = 9
        teacher_model = BertTokenClassifier(teacher_bert, num_class=cur_num_class).to(device)
        student_model = BertTokenClassifier(student_bert, num_class=cur_num_class).to(device)
    elif task in GLUE_TASKS:
        cur_num_class = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
        teacher_model = BertClassifier(teacher_bert, num_class=cur_num_class).to(device)
        student_model = BertClassifier(student_bert, num_class=cur_num_class).to(device)
    else:
        raise ValueError("Task is not defined. Must be one of pos, ner, imdb")

    teacher_model.load_state_dict(torch.load(teacher_name, map_location=device))
    student_model.load_state_dict(torch.load(student_name, map_location=device))
    print('Load', student_name)

    if task == 'pos' or task == 'ner':
        train_dataloader, valid_dataloader, test_dataloader = load_token_data(tokenizer, task, batch_size, start_i=0.0, end_i=0.5)

    elif task in GLUE_TASKS:
        train_dataloader, valid_dataloader, test_dataloader = load_glue_data(tokenizer, task, batch_size, start_i=0.0, end_i=0.5)
    else:
        raise ValueError("Task is not defined. Must be one of pos, ner, imdb")

    x = []
    sx = []
    sy = []
    ty = []

    vec = torch.tensor(vec).to(device)
    key = torch.tensor(key).to(device)
    skey = torch.tensor(skey).to(device)

    for batch in tqdm(train_dataloader):
        with torch.no_grad():
            if task == 'pos' or task == 'ner':
                text = batch['ids'].to(device)
                tags = batch['labels'].to(device)
                teacher_output = softmax_signal_wm(teacher_model(text), text, task, vec, key, skey, k, epsilon, num_classes=cur_num_class, shape='cosine', padding=0.0, device=device, wmidx=wmidx, sub=sub)
                s_output = student_model(text)
            elif task in GLUE_TASKS:
                input_id = batch['ids'].squeeze(1).to(device)
                mask = batch['atts'].to(device)
                tags = batch['labels'].to(device)
                teacher_output = softmax_signal_wm(teacher_model(input_id, mask), input_id, task, vec, key, skey, k, epsilon, num_classes=cur_num_class, shape='cosine', padding=0.0, device=device, tid=tid, wmidx=wmidx, sub=sub)
                text = input_id[:, tid]
                s_output = student_model(input_id, mask)

            else:
                raise ValueError("Task is not defined. Must be one of pos, ner, imdb")

            s_output = s_output.reshape(-1, cur_num_class)
            student_output = torch.softmax(s_output, dim=1)
            cur_x = torch.matmul(vec[text.reshape(-1, 1)], key)
            cur_x = torch.distributions.Normal(0, 1).cdf(cur_x / np.sqrt(key.shape[0] / 3)).reshape(-1)
            cur_sx = torch.matmul(vec[text.reshape(-1, 1)], skey)
            cur_sx = torch.distributions.Normal(0, 1).cdf(cur_sx / np.sqrt(key.shape[0] / 3)).reshape(-1)
            cur_tags = tags.reshape(-1)
            x += cur_x[cur_tags == target_class].tolist()
            sx += cur_sx[cur_tags == target_class].tolist()
            sy += teacher_output[cur_tags == target_class][:, target_class].tolist()
            ty += student_output[cur_tags == target_class][:, target_class].tolist()

    num_points_test = len(x)
    idx_high_score = torch.topk(torch.tensor(ty), int(num_points_test * ratio), dim=0).indices.sort().values
    idx_new = idx_high_score.tolist()
    x_new = np.array(x)[idx_new]
    sx_new = np.array(sx)[idx_new]
    sy_new = np.array(sy)[idx_new]
    ty_new = np.array(ty)[idx_new]
    xy_array = np.stack((x_new, ty_new, sy_new), axis=1)[sx_new < sub]
    freqs_array, _ = build_periodogram(xy_array, k=k)
    win005, psnr = get_spectrum_window(freqs_array[:, 0], freqs_array[:, 1], k, halfwidth=0.005)
    print('psnr:', psnr)
    return psnr


def test_deepjudge(teacher_name, student_name, device):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    teacher_bert = BertModel.from_pretrained('bert-base-uncased')
    student_bert = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device(device)
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    task = teacher_name.split("/")[2].split("_")[0]
    batch_size = 32
    if 'wm-1' in student_name:
        tseed = int(student_name.split('tseed-')[1][:2])
        nseed = int(student_name.split('nseed-')[1][:2])
    else:
        tseed = 11222
        nseed = 11222

    if task == 'pos':
        target_class = 22
        wmidx = [22]
        ratio = 0.98
    elif task == 'ner':
        target_class = 2
        wmidx = [2]
        ratio = 0.98
    elif task == 'mrpc':
        target_class = 0
        wmidx = [0]
        ratio = 0.95
    elif task == 'sst2':
        target_class = 0
        wmidx = [0]
        ratio = 0.98
    tid = 2
    sub = 0.5
    k = 16
    epsilon = 0.2
    torch.manual_seed(tseed)
    np.random.seed(nseed)

    key = np.random.rand(128)
    vec = np.random.randn(30522, 128)
    skey = np.load('select_key.npy')

    if task == 'pos':
        cur_num_class = 47
        teacher_model = BertTokenClassifier(teacher_bert, num_class=cur_num_class).to(device)
        student_model = BertTokenClassifier(student_bert, num_class=cur_num_class).to(device)
    elif task == 'ner':
        cur_num_class = 9
        teacher_model = BertTokenClassifier(teacher_bert, num_class=cur_num_class).to(device)
        student_model = BertTokenClassifier(student_bert, num_class=cur_num_class).to(device)
    elif task in GLUE_TASKS:
        cur_num_class = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
        teacher_model = BertClassifier(teacher_bert, num_class=cur_num_class).to(device)
        student_model = BertClassifier(student_bert, num_class=cur_num_class).to(device)
    else:
        raise ValueError("Task is not defined. Must be one of pos, ner, imdb")

    teacher_model.load_state_dict(torch.load(teacher_name, map_location=device))
    student_model.load_state_dict(torch.load(student_name, map_location=device))
    print('Load', student_name)

    if task == 'pos' or task == 'ner':
        train_dataloader, valid_dataloader, test_dataloader = load_token_data(tokenizer, task, batch_size, start_i=0.0, end_i=0.5)

    elif task in GLUE_TASKS:
        train_dataloader, valid_dataloader, test_dataloader = load_glue_data(tokenizer, task, batch_size, start_i=0.0, end_i=0.5)
    else:
        raise ValueError("Task is not defined. Must be one of pos, ner, imdb")

    x = []
    sx = []
    sy = []
    ty = []

    vec = torch.tensor(vec).to(device)
    key = torch.tensor(key).to(device)
    skey = torch.tensor(skey).to(device)
    DIGISTS = 10
    N = cur_num_class
    t_output_all = torch.empty((1, N))
    s_output_all = torch.empty((1, N))

    for batch in tqdm(train_dataloader):
        with torch.no_grad():
            if task == 'pos' or task == 'ner':
                text = batch['ids'].to(device)
                tags = batch['labels'].to(device)
                teacher_output = softmax_signal_wm(teacher_model(text), text, task, vec, key, skey, k, epsilon, num_classes=cur_num_class, shape='cosine', padding=0.0, device=device, wmidx=wmidx, sub=sub)
                s_output = student_model(text)
            elif task in GLUE_TASKS:
                input_id = batch['ids'].squeeze(1).to(device)
                mask = batch['atts'].to(device)
                tags = batch['labels'].to(device)
                teacher_output = softmax_signal_wm(teacher_model(input_id, mask), input_id, task, vec, key, skey, k, epsilon, num_classes=cur_num_class, shape='cosine', padding=0.0, device=device, tid=tid, wmidx=wmidx, sub=sub)
                text = input_id[:, tid]
                s_output = student_model(input_id, mask)

            else:
                raise ValueError("Task is not defined. Must be one of pos, ner, imdb")

            s_output = s_output.reshape(-1, cur_num_class)
            student_output = torch.softmax(s_output, dim=1)
            s_output_all = torch.cat([s_output_all, student_output.cpu()])
            t_output_all = torch.cat([t_output_all, teacher_output.cpu()])

    vectors1 = t_output_all[1:]
    vectors2 = s_output_all[1:]
    mid = (vectors1 + vectors2) / 2
    distances = (scipy.stats.entropy(vectors1, mid, axis=1) + scipy.stats.entropy(vectors2, mid, axis=1)) / 2
    dis = round(np.average(distances), DIGISTS)
    print('Distance:', dis)
    return dis


# model_list = sorted(glob.glob("./output/*"))
model_list = sorted(glob.glob("./new/*"))
res = {}
for m in model_list:
    model_name = m.split("/")[2]
    if 'teacher' in model_name:
        continue
    if 'ner' in model_name and 'pt' in model_name:
        if 'wm' in model_name or 'ori' in model_name:
            psnr = test_detect('./output/ner_teacher.pt', m, 'cuda:0')
            res[model_name] = psnr
# for m in model_list:
#     model_name = m.split("/")[2]
#     if 'teacher' in model_name:
#         continue
#     if 'mrpc' in model_name and 'pt' in model_name:
#         if 'wm' in model_name or 'ori' in model_name:
#             psnr = test_detect('./output/mrpc_teacher.pt', m, 'cuda:1')
#             res[model_name] = psnr
#     if 'sst2' in model_name and 'pt' in model_name:
#         if 'wm' in model_name or 'ori' in model_name:
#             psnr = test_detect('./output/sst2_teacher.pt', m, 'cuda:1')
#             res[model_name] = psnr
#     if 'ner' in model_name and 'pt' in model_name:
#         if 'wm' in model_name or 'ori' in model_name:
#             psnr = test_detect('./output/ner_teacher.pt', m, 'cuda:1')
#             res[model_name] = psnr
#     if 'pos' in model_name and 'pt' in model_name:
#         if 'wm' in model_name or 'ori' in model_name:
#             psnr = test_detect('./output/pos_teacher.pt', m, 'cuda:1')
#             res[model_name] = psnr
# for m in model_list:
#     model_name = m.split("/")[2]
#     if 'teacher' in model_name:
#         continue
#     if 'mrpc' in model_name and 'pt' in model_name:
#         if 'wm' in model_name or 'ori' in model_name:
#             psnr = test_deepjudge('./output/mrpc_teacher.pt', m, 'cuda:0')
#             res[model_name] = psnr
#     if 'sst2' in model_name and 'pt' in model_name:
#         if 'wm' in model_name or 'ori' in model_name:
#             psnr = test_deepjudge('./output/sst2_teacher.pt', m, 'cuda:0')
#             res[model_name] = psnr
#     if 'ner' in model_name and 'pt' in model_name:
#         if 'wm' in model_name or 'ori' in model_name:
#             psnr = test_deepjudge('./output/ner_teacher.pt', m, 'cuda:0')
#             res[model_name] = psnr
#     if 'pos' in model_name and 'pt' in model_name:
#         if 'wm' in model_name or 'ori' in model_name:
#             psnr = test_deepjudge('./output/pos_teacher.pt', m, 'cuda:0')
#             res[model_name] = psnr

print('FINISH!')
json.dump(res, open("result_noise.json", 'w'))
