import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

import wandb
wandb.login()

from pathlib import Path
import urllib.request

_BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_PATH = _BASE_DIR / "data" / "names.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"

# Create the data directory if it doesn't exist
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# Download the file if it doesn't exist locally
if not DATA_PATH.exists():
    print(f"Downloading {DATA_URL} to {DATA_PATH}...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Download complete.")

words = DATA_PATH.read_text(encoding="utf-8").splitlines()

words[:10]

len(words)

b = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1

chars = sorted(b.items(), key = lambda kv: -kv[1])

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))

stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0

itos = {i: ch for ch, i in stoi.items()}

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')

for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')

p = N[0].float()
p = p/p.sum()
p

g = torch.Generator().manual_seed(2147483647)

for _ in range(10):
  out = []
  ix = 0
  while True:
    p = N[ix].float()
    p = p/p.sum()
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))

# 1 ekleyerek model smoothing (yumuşatma) yapıyoruz
P = (N+1).float()

# Broadcasting sayesinde tüm matrisi tek seferde normalize ediyoruz:
# P = 27x27
# P.sum(1, keepdim=True) = 27x1
P /= P.sum(1, keepdim=True)

print(f"P shape: {P.shape}")

g = torch.Generator().manual_seed(2147483647)

for _ in range(10):
  out = []
  ix = 0
  while True:
    # Artık direkt P matrisinden ilgili satırı çekiyoruz
    p = P[ix]

    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))



# Örnek olarak ilk 3 kelimenin olasılıklarına ve log-olasılıklarına bakalım:
log_likelihood = 0.0
n = 0

for w in words[:3]:
  print(f"--- {w} ---")
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    print(f'{ch1}{ch2}: olasılık={prob:.4f}, log-olasılık={logprob:.4f}')

print('='*20)
print(f'log_likelihood = {log_likelihood:.4f}')
nll = -log_likelihood
print(f'negatif log_likelihood = {nll:.4f}')
print(f'Ortalama Kayıp (Loss / NLL) = {nll/n:.4f}')

# Tüm veri seti için NLL (Loss) hesaplama
log_likelihood = 0.0
n = 0

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1

nll = -log_likelihood
print(f'Tüm veri seti için Ortalama Kayıp (Loss/NLL) = {nll/n:.4f}')

# "andrejq" kelimesini test edelim
log_likelihood = 0.0
n = 0
for w in ["andrejq"]:
  print(f"--- {w} ---")
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    print(f'{ch1}{ch2}: olasılık={prob:.4f}, log-olasılık={logprob:.4f}')

nll = -log_likelihood
print(f'Ortalama Kayıp = {nll/n:.4f}')
print("\n'jq' geçişine dikkat! Olasılığı 0 değil, +1 smoothing sayesinde çok küçük de olsa bir olasılığı var ve loss sonsuz çıkmıyor.")

# Sinir ağı için eğitim setini (training set) oluşturalım
xs, ys = [], []

for w in words[:1]: # Şimdilik sadece ilk kelime ("emma") üzerinden görelim
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    print(f"{ch1} ---> {ch2}")
    xs.append(ix1)
    ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print("\nGirdiler (xs):", xs)
print("Hedefler (ys):", ys)

import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

# xs tensorünü one-hot vektörlere çevirme
# 27 farklı karakterimiz olduğu için num_classes=27
xenc = F.one_hot(xs, num_classes=27).float()

print("xenc boyutu (shape):", xenc.shape)
print("xenc veri tipi:", xenc.dtype)

# Görselleştirme (Sarı noktalar 1'leri, mor kısımlar 0'ları temsil eder)
plt.imshow(xenc)

# Ağırlık matrisini rastgele sayılarla başlatalım (Normal dağılım)
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)

# İleri yayılım (Forward pass): Girdiler x Ağırlıklar
# (5, 27) @ (27, 27) = (5, 27)
logits = xenc @ W

print("Ağırlık matrisi (W) boyutu:", W.shape)
print("Çıktı (logits) matrisi boyutu:", logits.shape)
print("\nİlk örneğimizin (ilk harfin) logits değerleri:\n", logits[0])

# 1. Adım: Skorları pozitife çevirmek için e tabanında üslerini alıyoruz (exp)
# Bu adım aslında istatistiksel modeldeki 'N' (sayma/counts) matrisine benzer bir etki yaratır.
counts = logits.exp()

# 2. Adım: Her satırı kendi toplamına bölerek olasılık haline getiriyoruz (0-1 arası ve toplamı 1)
# Bu adım istatistiksel modeldeki 'P' matrisini (normalizasyon) oluşturmaya benzer.
probs = counts / counts.sum(1, keepdims=True)

print("Olasılık matrisinin boyutu (probs):", probs.shape)
print("\nİlk örneğimizin olasılık değerleri:\n", probs[0])
print("\nBu olasılıkların toplamı (1.0 olmalı):", probs[0].sum().item())

# 5 örneğimiz var
num_examples = 5

# PyTorch'un indeksleme yeteneğini kullanarak doğru harflerin olasılıklarını çekiyoruz
# probs[0, 5], probs[1, 13], probs[2, 13] ... gibi
correct_probs = probs[torch.arange(num_examples), ys]
print("Doğru harflere verilen olasılıklar:", correct_probs)

# Logaritmalarını alıyoruz
log_probs = torch.log(correct_probs)

# Negatif ortalamasını (NLL) alarak Loss değerini buluyoruz
loss = -log_probs.mean()
print(f"\nLoss (Kayıp) değerimiz: {loss.item():.4f}")

get_ipython().system('jupyter nbconvert --to html /content/make_more.ipynb')


