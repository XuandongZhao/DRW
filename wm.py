import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def train_teacher(model, iterator, optimizer, scheduler, criterion, task, device):
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    epoch_loss = 0
    epoch_pred = []
    epoch_label = []
    model.train()
    for batch in tqdm(iterator):
        optimizer.zero_grad()
        label = batch['labels'].to(device)
        if task == 'imdb':
            input_id = batch['ids']['input_ids'].squeeze(1).to(device)
            mask = batch['ids']['attention_mask'].to(device)
            pred = model(input_id, mask)
        elif task == 'pos' or task == 'ner':
            text = batch['ids'].to(device)
            pred = model(text)
        elif task in GLUE_TASKS:
            input_id = batch['ids'].squeeze(1).to(device)
            mask = batch['atts'].to(device)
            pred = model(input_id, mask)
        else:
            raise ValueError("Task is not defined. Must be one of pos, ner, imdb, glue")

        pred = pred.view(-1, pred.shape[-1])
        label = label.view(-1)
        loss = criterion(pred, label)

        epoch_label += label.detach().cpu().tolist()
        epoch_pred += pred.argmax(dim=-1).detach().cpu().tolist()
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
    return epoch_loss / len(iterator), epoch_pred, epoch_label


def train_student_soft(student_model, teacher_model, iterator, optimizer, scheduler, criterion, task, num_class, device):
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    epoch_loss = 0
    epoch_pred = []
    epoch_label = []
    student_model.train()
    for batch in tqdm(iterator):
        optimizer.zero_grad()
        true_label = batch['labels'].to(device)
        if task == 'imdb':
            input_id = batch['ids']['input_ids'].squeeze(1).to(device)
            mask = batch['ids']['attention_mask'].to(device)
            pred = student_model(input_id, mask)
            with torch.no_grad():
                label = torch.softmax(teacher_model(input_id, mask), dim=-1)
        elif task == 'pos' or task == 'ner':
            text = batch['ids'].to(device)
            pred = student_model(text)
            with torch.no_grad():
                label = torch.softmax(teacher_model(text), dim=-1)
        elif task in GLUE_TASKS:
            input_id = batch['ids'].squeeze(1).to(device)
            mask = batch['atts'].to(device)
            pred = student_model(input_id, mask)
            with torch.no_grad():
                label = torch.softmax(teacher_model(input_id, mask), dim=-1)
        else:
            raise ValueError("Task is not defined. Must be one of pos, ner, imdb, glue")
        pred = pred.view(-1, pred.shape[-1])
        label = label.view(-1, pred.shape[-1])
        true_label = true_label.view(-1)
        loss = criterion(pred, label, true_label)

        epoch_label += label.argmax(dim=-1)[true_label != -100].detach().cpu().tolist()
        epoch_pred += pred.argmax(dim=-1)[true_label != -100].detach().cpu().tolist()
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
    return epoch_loss / len(iterator), epoch_pred, epoch_label


def train_student_hard(student_model, teacher_model, iterator, optimizer, scheduler, criterion, task, num_class, device):
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

    epoch_loss = 0
    epoch_pred = []
    epoch_label = []
    student_model.train()
    for batch in tqdm(iterator):
        optimizer.zero_grad()
        true_label = batch['labels'].to(device)
        if task == 'imdb':
            input_id = batch['ids']['input_ids'].squeeze(1).to(device)
            mask = batch['ids']['attention_mask'].to(device)
            pred = student_model(input_id, mask)
            with torch.no_grad():
                label = torch.softmax(teacher_model(input_id, mask), dim=-1)
        elif task == 'pos' or task == 'ner':
            text = batch['ids'].to(device)
            pred = student_model(text)
            with torch.no_grad():
                label = torch.softmax(teacher_model(text), dim=-1)
        elif task in GLUE_TASKS:
            input_id = batch['ids'].squeeze(1).to(device)
            mask = batch['atts'].to(device)
            pred = student_model(input_id, mask)
            with torch.no_grad():
                label = torch.softmax(teacher_model(input_id, mask), dim=-1)
        else:
            raise ValueError("Task is not defined. Must be one of pos, ner, imdb, glue")

        pred = pred.view(-1, pred.shape[-1])
        label = label.view(-1, label.shape[-1])
        true_label = true_label.view(-1)

        hard_label = torch.ones(label.shape[0], dtype=torch.int64).to(device)
        for idx in range(label.shape[0]):
            p = label[idx].cpu().numpy()
            p /= p.sum()  # overcome the tolerance
            hard_label[idx] = np.random.choice(num_class, p=p)

        loss = criterion(pred, hard_label)

        epoch_label += hard_label[true_label != -100].detach().cpu().tolist()
        epoch_pred += pred.argmax(dim=-1)[true_label != -100].detach().cpu().tolist()
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
    return epoch_loss / len(iterator), epoch_pred, epoch_label


def softmax_signal_wm(output, embs, task, vec, key, skey, k, epsilon, num_classes=18, shape='cosine',
                      padding=0.0, device='cpu', tid=0, wmidx=[], sub=1.0):
    N = num_classes
    if task == 'pos' or task == 'ner':
        output = output.reshape(-1, num_classes)
        x = torch.matmul(vec[embs.reshape(-1, 1)], key)
        sx = torch.matmul(vec[embs.reshape(-1, 1)], skey)
    else:
        x = torch.matmul(vec[embs[:, tid].reshape(-1, 1)], key)
        sx = torch.matmul(vec[embs[:, tid].reshape(-1, 1)], skey)
    x = torch.distributions.Normal(0, 1).cdf(x / np.sqrt(key.shape[0] / 3))
    sx = torch.distributions.Normal(0, 1).cdf(sx / np.sqrt(skey.shape[0] / 3))

    batch_size = output.shape[0]
    idx = torch.arange(batch_size)
    argm = torch.argmax(output, dim=1)
    if shape == 'cosine':
        phases = torch.tensor([np.pi for i in range(N * batch_size)]).view(-1, N).to(device)
        phases[idx, argm] = 0
        phi = torch.cos(k * x + phases)
    else:
        print('softmax_signal -- Invalid shape: returning zero signal')
        phi = 0

    epsilons = torch.tensor([epsilon / (N - 1) for i in range(N * batch_size)]).view(-1, N).to(device)
    epsilons[idx, argm] = epsilon

    sm = F.softmax(output, dim=1)
    smsigned = (sm + padding + 1e-25 + epsilons * (1 + phi)) / (1 + (torch.sum(epsilons[0]) + padding + 1e-25))

    if wmidx != []:
        selected = torch.isin(argm, torch.tensor(wmidx).to(device)) & (sx.reshape(-1) < sub)
        sm[selected] = 0
        smsigned[~selected] = 0
        smsigned += sm
    return smsigned


def train_student_soft_wm(student_model, teacher_model, vec, key, skey, iterator, optimizer, scheduler, criterion, task, num_class,
                          device, k=0.5, epsilon=0.2, tid=0, wmidx=[], sub=1.0):
    epoch_loss = 0
    epoch_pred = []
    epoch_label = []
    vec = torch.tensor(vec).to(device)
    key = torch.tensor(key).to(device)
    skey = torch.tensor(skey).to(device)
    student_model.train()
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

    for batch in tqdm(iterator):
        optimizer.zero_grad()
        true_label = batch['labels'].to(device)
        if task == 'imdb':
            input_id = batch['ids']['input_ids'].squeeze(1).to(device)
            mask = batch['ids']['attention_mask'].to(device)
            pred = student_model(input_id, mask)
            with torch.no_grad():
                # label = torch.softmax(teacher_model(input_id, mask), dim=-1)
                label = softmax_signal_wm(teacher_model(input_id, mask), input_id, task, vec, key, skey, k, epsilon, num_classes=num_class,
                                          shape='cosine', padding=0.0, device=device, tid=tid, wmidx=wmidx, sub=sub)
        elif task == 'pos' or task == 'ner':
            text = batch['ids'].to(device)
            pred = student_model(text)
            with torch.no_grad():
                # label = torch.softmax(teacher_model(text), dim=-1)
                label = softmax_signal_wm(teacher_model(text), text, task, vec, key, skey, k, epsilon, num_classes=num_class,
                                          shape='cosine', padding=0.0, device=device, wmidx=wmidx, sub=sub)
        elif task in GLUE_TASKS:
            input_id = batch['ids'].squeeze(1).to(device)
            mask = batch['atts'].to(device)
            pred = student_model(input_id, mask)
            with torch.no_grad():
                label = softmax_signal_wm(teacher_model(input_id, mask), input_id, task, vec, key, skey, k, epsilon, num_classes=num_class,
                                          shape='cosine', padding=0.0, device=device, tid=tid, wmidx=wmidx, sub=sub)
        else:
            raise ValueError("Task is not defined. Must be one of pos, ner, imdb, glue")

        pred = pred.view(-1, pred.shape[-1])
        label = label.view(-1, pred.shape[-1])
        true_label = true_label.view(-1)
        loss = criterion(pred, label, true_label)

        epoch_label += label.argmax(dim=-1)[true_label != -100].detach().cpu().tolist()
        epoch_pred += pred.argmax(dim=-1)[true_label != -100].detach().cpu().tolist()
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
    return epoch_loss / len(iterator), epoch_pred, epoch_label


def train_student_hard_wm(student_model, teacher_model, vec, key, skey, iterator, optimizer, scheduler, criterion, task, num_class,
                          device, k=0.5, epsilon=0.2, tid=0, wmidx=[], sub=1.0):
    epoch_loss = 0
    epoch_pred = []
    epoch_label = []
    vec = torch.tensor(vec).to(device)
    key = torch.tensor(key).to(device)
    skey = torch.tensor(skey).to(device)
    student_model.train()
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    total_i = len(iterator)

    for batch in tqdm(iterator):
        optimizer.zero_grad()
        true_label = batch['labels'].to(device)
        if task == 'imdb':
            input_id = batch['ids']['input_ids'].squeeze(1).to(device)
            mask = batch['ids']['attention_mask'].to(device)
            pred = student_model(input_id, mask)
            with torch.no_grad():
                # label = torch.softmax(teacher_model(input_id, mask), dim=-1)
                label = softmax_signal_wm(teacher_model(input_id, mask), input_id, task, vec, key, skey, k, epsilon, num_classes=num_class,
                                          shape='cosine', padding=0.0, device=device, tid=tid, wmidx=wmidx, sub=sub)
        elif task == 'pos' or task == 'ner':
            text = batch['ids'].to(device)
            pred = student_model(text)
            with torch.no_grad():
                # label = torch.softmax(teacher_model(text), dim=-1)
                label = softmax_signal_wm(teacher_model(text), text, task, vec, key, skey, k, epsilon, num_classes=num_class,
                                          shape='cosine', padding=0.0, device=device, wmidx=wmidx, sub=sub)
        elif task in GLUE_TASKS:
            input_id = batch['ids'].squeeze(1).to(device)
            mask = batch['atts'].to(device)
            pred = student_model(input_id, mask)
            with torch.no_grad():
                label = softmax_signal_wm(teacher_model(input_id, mask), input_id, task, vec, key, skey, k, epsilon, num_classes=num_class,
                                          shape='cosine', padding=0.0, device=device, tid=tid, wmidx=wmidx, sub=sub)
        else:
            raise ValueError("Task is not defined. Must be one of pos, ner, imdb, glue")

        pred = pred.view(-1, pred.shape[-1])
        label = label.view(-1, label.shape[-1])
        true_label = true_label.view(-1)
        hard_label = torch.ones(label.shape[0], dtype=torch.int64).to(device)
        for idx in range(label.shape[0]):
            p = label[idx].cpu().numpy()
            p /= p.sum()  # overcome the tolerance
            hard_label[idx] = np.random.choice(num_class, p=p)

        loss = criterion(pred, hard_label)

        epoch_label += hard_label[true_label != -100].detach().cpu().tolist()
        epoch_pred += pred.argmax(dim=-1)[true_label != -100].detach().cpu().tolist()
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
    return epoch_loss / len(iterator), epoch_pred, epoch_label


def evaluate(model, iterator, criterion, task, device):
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

    epoch_loss = 0
    epoch_pred = []
    epoch_label = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator):
            label = batch['labels'].to(device)
            if task == 'imdb':
                input_id = batch['ids']['input_ids'].squeeze(1).to(device)
                mask = batch['ids']['attention_mask'].to(device)
                pred = model(input_id, mask)
            elif task == 'pos' or task == 'ner':
                text = batch['ids'].to(device)
                pred = model(text)
            elif task in GLUE_TASKS:
                input_id = batch['ids'].squeeze(1).to(device)
                mask = batch['atts'].to(device)
                pred = model(input_id, mask)
            else:
                raise ValueError("Task is not defined. Must be one of pos, ner, imdb, glue")
            pred = pred.view(-1, pred.shape[-1])
            label = label.view(-1)
            loss = criterion(pred, label)

            epoch_label += label.detach().cpu().tolist()
            epoch_pred += pred.argmax(dim=-1).detach().cpu().tolist()
            epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_pred, epoch_label


def teacher_hard_acc(teacher_model, vec, key, skey, iterator, task, num_class, device, k=6.0, epsilon=0.2, tid=1, wmidx=[], sub=1.0):
    print('Hard version', sub)
    GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    epoch_loss = 0
    epoch_pred = []
    epoch_label = []
    vec = torch.tensor(vec).to(device)
    key = torch.tensor(key).to(device)
    skey = torch.tensor(skey).to(device)
    print('wmidx is', wmidx)
    for batch in tqdm(iterator):
        true_label = batch['labels'].to(device)
        with torch.no_grad():
            if task == 'imdb':
                input_id = batch['ids']['input_ids'].squeeze(1).to(device)
                mask = batch['ids']['attention_mask'].to(device)
                label = softmax_signal_wm(teacher_model(input_id, mask), input_id, task, vec, key, skey, k, epsilon, num_classes=num_class,
                                          shape='cosine', padding=0.0, device=device, tid=tid, wmidx=wmidx, sub=sub)
            elif task == 'pos' or task == 'ner':
                text = batch['ids'].to(device)
                label = softmax_signal_wm(teacher_model(text), text, task, vec, key, skey, k, epsilon, num_classes=num_class,
                                          shape='cosine', padding=0.0, device=device, wmidx=wmidx, sub=sub)
            elif task in GLUE_TASKS:
                input_id = batch['ids'].squeeze(1).to(device)
                mask = batch['atts'].to(device)
                label = softmax_signal_wm(teacher_model(input_id, mask), input_id, task, vec, key, skey, k, epsilon, num_classes=num_class,
                                          shape='cosine', padding=0.0, device=device, tid=tid, wmidx=wmidx, sub=sub)

        label = label.view(-1, label.shape[-1])
        true_label = true_label.view(-1)

        hard_label = torch.ones(label.shape[0], dtype=torch.int64).to(device)
        for idx in range(label.shape[0]):
            p = label[idx].cpu().numpy()
            p /= p.sum()  # overcome the tolerance
            hard_label[idx] = np.random.choice(num_class, p=p)

        epoch_pred += hard_label[true_label != -100].detach().cpu().tolist()
        epoch_label += true_label[true_label != -100].detach().cpu().tolist()

    return epoch_loss / len(iterator), epoch_pred, epoch_label
