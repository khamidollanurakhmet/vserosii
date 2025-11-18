# -*- coding: utf-8 -*-
"""
Примеры запуска:
- Классификация синтетики (k-mer + LogisticRegression):
  python baseline_dnk.py --mode classify --n 2000 --len 120 --k 3
- Генерация по целевому GC и мотиву (эвристический отжиг):
  python baseline_dnk.py --mode generate --len 150 --gc 0.6 --motif ACGT --count 10 --out gen.json
- Обучение простого LSTM LM на FASTA/списке строк:
  python baseline_dnk.py --mode lm_train --in sequences.fasta --epochs 3 --save dnk_lm.pt
- Семплирование из LM:
  python baseline_dnk.py --mode lm_sample --load dnk_lm.pt --len 200 --count 5

Зависимости: см. requirements_dnk.txt (PyTorch/biopython — опциональны).
"""
from __future__ import annotations

import os
import re
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---------------------- Конфиг ----------------------
@dataclass
class DNKConfig:
    seed: int = 42
    k: int = 3              # для k-mer признаков
    max_vocab: int = 4      # алфавит ДНК: A,C,G,T
    batch_size: int = 64
    epochs: int = 3
    lr: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 1
    save_dir: str = 'dnk_models'

CFG = DNKConfig()

# ---------------------- Утилиты ----------------------
NUC = ['A', 'C', 'G', 'T']
NUC_TO_ID = {c: i for i, c in enumerate(NUC)}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------- DNA генерация и утилиты ----------------------

def random_dna(length: int, p: Optional[List[float]] = None) -> str:
    if p is None:
        p = [0.25, 0.25, 0.25, 0.25]
    return ''.join(np.random.choice(NUC, size=length, p=p))


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    g = seq.count('G')
    c = seq.count('C')
    return (g + c) / len(seq)


def has_motif(seq: str, motif: str) -> bool:
    return motif in seq if motif else True


def insert_motif(seq: str, motif: str) -> str:
    if not motif:
        return seq
    pos = random.randrange(0, max(1, len(seq) - len(motif) + 1))
    return seq[:pos] + motif + seq[pos+len(motif):]


# ---------------------- Синтетический датасет для классификации ----------------------
# Класс 1: последовательности с более высоким GC и вшитым мотивом (с вероятностью),
# Класс 0: без мотива, с низким GC.

def make_synthetic_dataset(n: int = 2000, length: int = 100, motif: str = 'ACGT', gc_pos: float = 0.6, gc_neg: float = 0.4, p_motif: float = 0.8):
    set_seed(CFG.seed)
    X = []
    y = []
    for i in range(n):
        if i < n // 2:
            p = [(1 - gc_neg) / 2, gc_neg / 2, gc_neg / 2, (1 - gc_neg) / 2]  # A,C,G,T
            s = random_dna(length, p)
            y.append(0)
        else:
            p = [(1 - gc_pos) / 2, gc_pos / 2, gc_pos / 2, (1 - gc_pos) / 2]
            s = random_dna(length, p)
            if random.random() < p_motif and motif:
                s = insert_motif(s, motif)
            y.append(1)
        X.append(s)
    return X, np.array(y, dtype=int)


# ---------------------- k-mer признаки ----------------------

def all_kmers(k: int) -> List[str]:
    if k == 1:
        return NUC
    prev = all_kmers(k - 1)
    return [p + c for p in prev for c in NUC]


def kmer_counts(seqs: List[str], k: int) -> np.ndarray:
    vocab = all_kmers(k)
    idx = {km: i for i, km in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            km = s[j:j+k]
            if all(ch in NUC_TO_ID for ch in km):
                X[i, idx[km]] += 1.0
        # нормировка на длину окна
        if len(s) - k + 1 > 0:
            X[i] /= (len(s) - k + 1)
    return X


# ---------------------- Классификация: sklearn baseline ----------------------

def classify_baseline(n: int, length: int, k: int, motif: str):
    if not SKLEARN_AVAILABLE:
        print('scikit-learn не установлен. Установите из requirements_dnk.txt')
        return
    X_str, y = make_synthetic_dataset(n=n, length=length, motif=motif)
    X = kmer_counts(X_str, k)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=CFG.seed, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    print({'accuracy': float(acc), 'f1': float(f1)})


# ---------------------- Генерация по целям (GC/мотив): эвристика ----------------------
# Целевая функция: |GC(seq) - target_gc| + штраф за отсутствие мотива.
# Мутации: заменяем случайный нуклеотид; симулированный отжиг.

def score_seq(seq: str, target_gc: float, motif: Optional[str] = None, motif_penalty: float = 1.0) -> float:
    s = abs(gc_content(seq) - target_gc)
    if motif and motif not in seq:
        s += motif_penalty
    return s


def mutate(seq: str) -> str:
    if not seq:
        return seq
    i = random.randrange(len(seq))
    cur = seq[i]
    choices = [c for c in NUC if c != cur]
    return seq[:i] + random.choice(choices) + seq[i+1:]


def anneal_generate(length: int, target_gc: float, motif: str = '', iters: int = 2000, start: Optional[str] = None) -> str:
    if start is None:
        # старт по близкому GC
        p = [(1 - target_gc) / 2, target_gc / 2, target_gc / 2, (1 - target_gc) / 2]
        cur = random_dna(length, p)
    else:
        cur = start
    cur_sc = score_seq(cur, target_gc, motif)
    best, best_sc = cur, cur_sc
    for t in tqdm(range(iters), desc='anneal'):
        T = max(0.01, (1 - t / max(1, iters)) * 1.0)
        cand = mutate(cur)
        cand_sc = score_seq(cand, target_gc, motif)
        delta = cand_sc - cur_sc
        if delta < 0 or random.random() < math.exp(-delta / T):
            cur, cur_sc = cand, cand_sc
            if cand_sc < best_sc:
                best, best_sc = cand, cand_sc
    return best


# ---------------------- Простая LM на PyTorch (опционально) ----------------------
class DNKTokenizer:
    def __init__(self):
        self.stoi = {c: i for i, c in enumerate(NUC)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos.get(i, 'A') for i in ids)


if TORCH_AVAILABLE:
    class DNKDataset(Dataset):
        def __init__(self, seqs: List[str], ctx: int = 64):
            self.tok = DNKTokenizer()
            self.ctx = ctx
            self.data = []
            for s in seqs:
                ids = self.tok.encode(s)
                # нарезаем контекстные окна для LM
                for i in range(len(ids) - 1):
                    l = max(0, i - ctx + 1)
                    x = ids[l:i+1]
                    if len(x) < ctx:
                        x = [0] * (ctx - len(x)) + x
                    y = ids[i+1]
                    self.data.append((x, y))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x, y = self.data[idx]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    class SimpleLSTM(nn.Module):
        def __init__(self, vocab_size: int = 4, embed: int = 16, hidden: int = 128, layers: int = 1):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed)
            self.lstm = nn.LSTM(embed, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Linear(hidden, vocab_size)

        def forward(self, x):
            e = self.emb(x)
            out, _ = self.lstm(e)
            last = out[:, -1, :]
            logits = self.head(last)
            return logits

    def train_lm(seqs: List[str], save_path: str, epochs: int = 3, ctx: int = 64):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ds = DNKDataset(seqs, ctx=ctx)
        dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True)
        model = SimpleLSTM(vocab_size=4, embed=16, hidden=CFG.hidden_size, layers=CFG.num_layers).to(device)
        opt = optim.Adam(model.parameters(), lr=CFG.lr)
        crit = nn.CrossEntropyLoss()
        for ep in range(epochs):
            model.train()
            losses = []
            for xb, yb in tqdm(dl, desc=f'LM ep{ep+1}'):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print({'epoch': ep+1, 'loss': float(np.mean(losses))})
        ensure_dir(os.path.dirname(save_path) or '.')
        torch.save({'model': model.state_dict(), 'cfg': {'hidden': CFG.hidden_size, 'layers': CFG.num_layers}}, save_path)
        return model

    def sample_lm(load_path: str, length: int = 200, count: int = 1, ctx: int = 64, prompt: str = '') -> List[str]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(load_path, map_location=device)
        hidden = ckpt['cfg'].get('hidden', CFG.hidden_size)
        layers = ckpt['cfg'].get('layers', CFG.num_layers)
        model = SimpleLSTM(vocab_size=4, embed=16, hidden=hidden, layers=layers).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        tok = DNKTokenizer()
        start_ids = tok.encode(prompt) if prompt else []
        results = []
        with torch.no_grad():
            for _ in range(count):
                seq = start_ids[:]
                while len(seq) < length:
                    ctx_ids = seq[-ctx:]
                    if len(ctx_ids) < ctx:
                        ctx_ids = [0] * (ctx - len(ctx_ids)) + ctx_ids
                    x = torch.tensor([ctx_ids], dtype=torch.long, device=device)
                    logits = model(x)
                    prob = torch.softmax(logits[0], dim=0).cpu().numpy()
                    nxt = int(np.random.choice(np.arange(4), p=prob))
                    seq.append(nxt)
                results.append(tok.decode(seq[:length]))
        return results


# ---------------------- IO ----------------------

def read_fasta_or_lines(path: str) -> List[str]:
    seqs = []
    if not path or not os.path.exists(path):
        return seqs
    with open(path, 'r', encoding='utf-8') as f:
        buf = []
        is_fasta = None
        for line in f:
            line = line.strip().upper()
            if not line:
                continue
            if line.startswith('>'):
                is_fasta = True if is_fasta is None else is_fasta
                if buf:
                    seqs.append(''.join(buf))
                    buf = []
                continue
            if is_fasta is None:
                # если нет заголовков, трактуем как по строкам
                seqs.append(re.sub('[^ACGT]', '', line))
            else:
                buf.append(re.sub('[^ACGT]', '', line))
        if buf:
            seqs.append(''.join(buf))
    return [s for s in seqs if s]


def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



# ---------------------- CLI ----------------------

def parse_args():
    p = argparse.ArgumentParser(description='Бейзлайн ДНК: классификация, генерация, LM')
    p.add_argument('--mode', type=str, required=True, choices=['classify', 'generate', 'lm_train', 'lm_sample'])
    # классификация
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--len', type=int, default=120, dest='length')
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--motif', type=str, default='ACGT')
    # генерация
    p.add_argument('--gc', type=float, default=0.5)
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--out', type=str, default=None)
    # LM
    p.add_argument('--in', type=str, dest='inp', default=None)
    p.add_argument('--save', type=str, default='dnk_lm.pt')
    p.add_argument('--load', type=str, default=None)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--prompt', type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(CFG.seed)

    if args.mode == 'classify':
        classify_baseline(n=args.n, length=args.length, k=args.k, motif=args.motif)

    elif args.mode == 'generate':
        seqs = [anneal_generate(length=args.length, target_gc=args.gc, motif=args.motif, iters=2000) for _ in range(args.count)]
        res = [{'seq': s, 'gc': gc_content(s), 'has_motif': has_motif(s, args.motif)} for s in seqs]
        print({'generated': len(res), 'avg_gc': float(np.mean([r['gc'] for r in res]))})
        if args.out:
            save_json(res, args.out)
            print(f'Сохранено: {args.out}')

    elif args.mode == 'lm_train':
        if not TORCH_AVAILABLE:
            print('PyTorch не установлен. Установите из requirements_dnk.txt')
            return
        seqs = read_fasta_or_lines(args.inp)
        if not seqs:
            # fallback: синтетика
            seqs, _ = make_synthetic_dataset(n=1000, length=args.length, motif=args.motif)
        ensure_dir(CFG.save_dir)
        save_path = args.save if args.save else os.path.join(CFG.save_dir, 'dnk_lm.pt')
        train_lm(seqs, save_path=save_path, epochs=args.epochs)
        print(f'Сохранено LM в {save_path}')

    elif args.mode == 'lm_sample':
        if not TORCH_AVAILABLE:
            print('PyTorch не установлен. Установите из requirements_dnk.txt')
            return
        if not args.load or not os.path.exists(args.load):
            print('Не найден путь к модели (--load)')
            return
        seqs = sample_lm(load_path=args.load, length=args.length, count=args.count, prompt=args.prompt)
        res = [{'seq': s, 'gc': gc_content(s)} for s in seqs]
        print({'samples': len(res), 'avg_gc': float(np.mean([r['gc'] for r in res]))})
        if args.out:
            save_json(res, args.out)
            print(f'Сохранено: {args.out}')


if __name__ == '__main__':
    main()
