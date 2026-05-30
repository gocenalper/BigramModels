"""
Generate a visually polished English PDF report for the Bigram Models project.

The script reproduces the work done in `make_more.py` (statistical bigram model
+ single-layer neural network) end to end, captures real figures and metrics,
and lays everything out in a multi-page PDF using matplotlib's PdfPages backend
(no external PDF dependency required).

Usage:
    python generate_report.py
Output:
    report/BigramModels_Report.pdf
"""

from pathlib import Path
import urllib.request

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
INK = "#1b2a4a"        # deep navy for headings
ACCENT = "#3b6ea5"     # medium blue
ACCENT2 = "#e07a5f"    # warm coral for highlights
MUTED = "#5b6b80"      # muted slate for body text
LIGHT = "#eef3f8"      # light panel fill
GREEN = "#2f9e6f"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 150,
})

PAGE = (8.27, 11.69)  # A4 portrait in inches


# --------------------------------------------------------------------------- #
# Data + model (reproduces make_more.py)
# --------------------------------------------------------------------------- #
def load_words():
    base = Path(__file__).parent
    data_path = base / "data" / "names.txt"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    if not data_path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, data_path)
    return data_path.read_text(encoding="utf-8").splitlines()


def build_model(words):
    chars = sorted(list(set("".join(words))))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}

    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1

    # Smoothed probability matrix
    P = (N + 1).float()
    P /= P.sum(1, keepdim=True)
    return stoi, itos, N, P


def sample_names(P, itos, seed=2147483647, count=8):
    g = torch.Generator().manual_seed(seed)
    out_names = []
    for _ in range(count):
        out, ix = [], 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, 1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        out_names.append("".join(out))
    return out_names


def dataset_nll(words, stoi, P):
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
            n += 1
    return (-log_likelihood / n).item()


def neural_net_demo(words, stoi):
    """Single training example ('emma') through the 1-layer net, as in make_more.py."""
    xs, ys = [], []
    for w in words[:1]:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    xs, ys = torch.tensor(xs), torch.tensor(ys)
    xenc = F.one_hot(xs, num_classes=27).float()

    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g)
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(len(ys)), ys].log().mean()
    return xs, ys, xenc, W, probs, loss.item()


# --------------------------------------------------------------------------- #
# Page helpers
# --------------------------------------------------------------------------- #
def new_page():
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    return fig


def header_band(fig, kicker, title, page_no):
    """Top band with kicker + title and a footer with page number."""
    ax = fig.add_axes([0, 0.93, 1, 0.07])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                boxstyle="square,pad=0", color=INK,
                                transform=ax.transAxes, zorder=0))
    ax.text(0.06, 0.62, title, transform=ax.transAxes, color="white",
            fontsize=17, fontweight="bold", va="center")
    ax.text(0.06, 0.22, kicker, transform=ax.transAxes, color="#9db8d8",
            fontsize=9.5, va="center", fontweight="bold")
    # footer
    fax = fig.add_axes([0, 0, 1, 0.035])
    fax.axis("off")
    fax.text(0.06, 0.4, "Bigram Language Models  ·  makemore study notes",
             color=MUTED, fontsize=7.5, va="center")
    fax.text(0.94, 0.4, f"{page_no}", color=MUTED, fontsize=8,
             va="center", ha="right", fontweight="bold")


def panel(fig, rect, facecolor=LIGHT):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=0, facecolor=facecolor,
                                transform=ax.transAxes))
    return ax


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_cover(pdf, stats):
    fig = new_page()
    # full-bleed navy top
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.62), 1, 0.38, color=INK, zorder=0))
    ax.add_patch(plt.Rectangle((0, 0.60), 1, 0.02, color=ACCENT2, zorder=1))

    ax.text(0.08, 0.86, "BIGRAM LANGUAGE MODELS", fontsize=30,
            fontweight="bold", color="white")
    ax.text(0.08, 0.805, "From Counting to a Neural Network", fontsize=16,
            color="#9db8d8")
    ax.text(0.08, 0.70, "A study-notes report on building a character-level\n"
                        "name generator, following Andrej Karpathy's makemore.",
            fontsize=12, color="#d7e2f0")

    # stat chips
    chips = [
        (f"{stats['n_words']:,}", "names in dataset"),
        ("27", "token vocabulary (a–z + .)"),
        (f"{stats['nll']:.3f}", "avg negative log-likelihood"),
        (f"{stats['n_bigrams']:,}", "bigram transitions counted"),
    ]
    x0 = 0.08
    for i, (big, small) in enumerate(chips):
        x = x0 + (i % 2) * 0.45
        y = 0.45 - (i // 2) * 0.13
        a = fig.add_axes([x, y, 0.40, 0.11])
        a.axis("off")
        a.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                   boxstyle="round,pad=0.02,rounding_size=0.08",
                                   facecolor=LIGHT, edgecolor=ACCENT, linewidth=1.2,
                                   transform=a.transAxes))
        a.text(0.07, 0.60, big, fontsize=22, fontweight="bold", color=ACCENT,
               transform=a.transAxes)
        a.text(0.07, 0.22, small, fontsize=10, color=MUTED, transform=a.transAxes)

    ax.text(0.08, 0.10, "Author: alpergocen35", fontsize=11, color=INK,
            fontweight="bold")
    ax.text(0.08, 0.065, "Repository: BigramModels", fontsize=10, color=MUTED)
    ax.text(0.92, 0.065, "Generated 2026", fontsize=10, color=MUTED, ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def page_overview(pdf, stats, sample_first10):
    fig = new_page()
    header_band(fig, "SECTION 01", "Project Overview", 2)

    ax = fig.add_axes([0.06, 0.50, 0.88, 0.40])
    ax.axis("off")
    body = (
        "This project re-implements a character-level language model that "
        "generates new, name-like words. The model learns the statistics of "
        "which letter tends to follow which, then samples fresh names one "
        "character at a time.\n\n"
        "Two complementary approaches are explored side by side:\n\n"
        "   1.  A statistical bigram model that simply counts how often each "
        "pair of adjacent characters occurs, then normalises those counts into "
        "probabilities.\n\n"
        "   2.  A single-layer neural network that learns an equivalent "
        "probability table through gradient-based optimisation, framing the "
        "exact same problem as machine learning.\n\n"
        "Both are evaluated with the same yardstick — the average negative "
        "log-likelihood (NLL) — so the two worlds can be compared directly. "
        "The dataset is the classic list of ~32k English first names used in "
        "Karpathy's makemore series."
    )
    ax.text(0, 1, body, fontsize=11.5, color=MUTED, va="top", linespacing=1.5,
            wrap=True)

    # Pipeline strip
    pax = panel(fig, [0.06, 0.30, 0.88, 0.15])
    steps = ["Names\ndataset", "Count\nbigrams", "Probability\nmatrix P",
             "Sample\nnew names", "Evaluate\nNLL loss"]
    n = len(steps)
    for i, s in enumerate(steps):
        cx = (i + 0.5) / n
        pax.text(cx, 0.62, s, ha="center", va="center", fontsize=10,
                 fontweight="bold", color=INK, transform=pax.transAxes)
        if i < n - 1:
            pax.annotate("", xy=((i + 1) / n - 0.01, 0.62),
                         xytext=((i + 1) / n - 0.04, 0.62),
                         xycoords=pax.transAxes,
                         arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=2))
    pax.text(0.5, 0.18, "End-to-end pipeline", ha="center", fontsize=9,
             style="italic", color=MUTED, transform=pax.transAxes)

    # Sample preview
    sax = panel(fig, [0.06, 0.07, 0.88, 0.18], facecolor="#fdf3ef")
    sax.text(0.04, 0.82, "First names in the dataset", fontsize=10.5,
             fontweight="bold", color=ACCENT2, transform=sax.transAxes)
    sax.text(0.04, 0.45, "   ".join(sample_first10[:7]), fontsize=13,
             color=INK, transform=sax.transAxes, family="DejaVu Sans Mono")
    sax.text(0.04, 0.16, "Each name is wrapped with a special '.' boundary "
                         "token marking where a word starts and ends.",
             fontsize=9, color=MUTED, transform=sax.transAxes, style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def page_heatmap(pdf, N, itos):
    fig = new_page()
    header_band(fig, "SECTION 02", "The Bigram Count Matrix", 3)

    ax = fig.add_axes([0.06, 0.82, 0.88, 0.08])
    ax.axis("off")
    ax.text(0, 0.5, "Every cell counts how often the row-character is "
                    "immediately followed by the column-character across all "
                    "32k names. Brighter = more frequent. This 27x27 table is "
                    "the entire 'knowledge' of the statistical model.",
            fontsize=10.5, color=MUTED, va="center", wrap=True, linespacing=1.4)

    hax = fig.add_axes([0.07, 0.10, 0.86, 0.68])
    Nn = N.numpy()
    hax.imshow(Nn, cmap="Blues")
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            hax.text(j, i, chstr, ha="center", va="bottom", color="gray",
                     fontsize=5.0)
            hax.text(j, i, Nn[i, j], ha="center", va="top", color="gray",
                     fontsize=4.6)
    hax.axis("off")
    hax.set_title("Bigram frequency heatmap (rows = current char, "
                  "cols = next char)", fontsize=10, color=INK, pad=8)
    pdf.savefig(fig)
    plt.close(fig)


def page_distribution(pdf, N, P, itos):
    fig = new_page()
    header_band(fig, "SECTION 03", "Reading the Statistics", 4)

    # Top: which letters start a name (row 0 = '.')
    ax1 = fig.add_axes([0.10, 0.56, 0.82, 0.30])
    start = P[0, 1:].numpy()
    letters = [itos[i] for i in range(1, 27)]
    bars = ax1.bar(letters, start, color=ACCENT)
    top = int(np.argmax(start))
    bars[top].set_color(ACCENT2)
    ax1.set_title("Probability of each letter STARTING a name",
                  fontsize=11, color=INK, fontweight="bold")
    ax1.set_ylabel("P(first letter)", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=8)
    ax1.text(top, start[top], f"  '{letters[top]}' most likely",
             color=ACCENT2, fontsize=9, fontweight="bold", va="bottom")

    # Bottom: which letters most often end a name (col 0 = '.')
    ax2 = fig.add_axes([0.10, 0.14, 0.82, 0.30])
    end = P[1:, 0].numpy()
    bars2 = ax2.bar(letters, end, color=GREEN)
    tope = int(np.argmax(end))
    bars2[tope].set_color(ACCENT2)
    ax2.set_title("Probability that a name ENDS right after each letter",
                  fontsize=11, color=INK, fontweight="bold")
    ax2.set_ylabel("P(end | letter)", fontsize=9)
    ax2.set_xlabel("letter", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)
    ax2.text(tope, end[tope], f"  '{letters[tope]}' most likely ending",
             color=ACCENT2, fontsize=9, fontweight="bold", va="bottom")

    fig.text(0.10, 0.90, "The probability matrix P is just the count matrix "
                         "normalised row-by-row (with +1 smoothing). Slicing it "
                         "reveals interpretable structure:",
             fontsize=10.5, color=MUTED, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def page_sampling_nll(pdf, samples, nll, n_params):
    fig = new_page()
    header_band(fig, "SECTION 04", "Sampling & Evaluation", 5)

    # Sampling panel
    sax = panel(fig, [0.06, 0.55, 0.88, 0.33], facecolor=LIGHT)
    sax.text(0.04, 0.90, "Names invented by the model", fontsize=12,
             fontweight="bold", color=INK, transform=sax.transAxes)
    sax.text(0.04, 0.80, "Starting from '.', repeatedly sample the next "
                         "character from P until '.' is drawn again:",
             fontsize=9.5, color=MUTED, transform=sax.transAxes, style="italic")
    cols = 2
    for i, name in enumerate(samples):
        cx = 0.07 + (i % cols) * 0.47
        cy = 0.62 - (i // cols) * 0.135
        sax.text(cx, cy, "•  " + name, fontsize=14, color=ACCENT,
                 transform=sax.transAxes, family="DejaVu Sans Mono",
                 fontweight="bold")
    sax.text(0.04, 0.05, "Not real names, but clearly name-like — the model has "
                         "captured plausible letter transitions.",
             fontsize=9, color=MUTED, transform=sax.transAxes, style="italic")

    # NLL explanation
    eax = fig.add_axes([0.06, 0.34, 0.88, 0.16])
    eax.axis("off")
    eax.text(0, 1, "How good is the model?  Negative Log-Likelihood (NLL)",
             fontsize=12, fontweight="bold", color=INK, va="top")
    eax.text(0, 0.62,
             "The quality of the model is measured by the average negative "
             "log-likelihood over every bigram in the dataset. A perfect model "
             "scores 0; lower is better. Equivalently, loss = -mean(log P(next | current)).",
             fontsize=10.5, color=MUTED, va="top", wrap=True, linespacing=1.45)

    # Big metric
    max = fig.add_axes([0.06, 0.12, 0.42, 0.18])
    max.axis("off")
    max.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                 boxstyle="round,pad=0.02,rounding_size=0.06",
                                 facecolor=INK, transform=max.transAxes))
    max.text(0.5, 0.66, f"{nll:.4f}", ha="center", fontsize=34,
             fontweight="bold", color="white", transform=max.transAxes)
    max.text(0.5, 0.24, "average NLL  (whole dataset)", ha="center",
             fontsize=10, color="#9db8d8", transform=max.transAxes)

    m2 = fig.add_axes([0.52, 0.12, 0.42, 0.18])
    m2.axis("off")
    m2.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=ACCENT2, transform=m2.transAxes))
    m2.text(0.5, 0.66, "+1", ha="center", fontsize=34, fontweight="bold",
            color="white", transform=m2.transAxes)
    max_label = ("Laplace smoothing added to every count so no transition has "
                 "zero probability — keeping the loss finite even for unseen "
                 "pairs like 'jq'.")
    m2.text(0.5, 0.20, "model smoothing", ha="center", fontsize=10,
            color="white", transform=m2.transAxes)
    fig.text(0.06, 0.085, max_label, fontsize=8.6, color=MUTED, style="italic",
             wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def page_neuralnet(pdf, xenc, W, probs, nn_loss):
    fig = new_page()
    header_band(fig, "SECTION 05", "The Neural Network View", 6)

    fig.text(0.06, 0.88,
             "The same problem, reframed as learning. Instead of counting, a "
             "single linear layer (27→27 weights) turns a one-hot encoded input "
             "character into a probability distribution over the next character "
             "via the softmax function.",
             fontsize=10.5, color=MUTED, wrap=True, linespacing=1.45)

    # forward pass diagram (text formula)
    dax = panel(fig, [0.06, 0.70, 0.88, 0.11], facecolor=LIGHT)
    dax.text(0.5, 0.5,
             "one-hot(x)  →  logits = x · W  →  counts = exp(logits)  "
             "→  probs = counts / Σcounts",
             ha="center", va="center", fontsize=9.5, color=INK,
             fontweight="bold", transform=dax.transAxes,
             family="DejaVu Sans Mono")

    # one-hot encoding of 'emma' example
    ax1 = fig.add_axes([0.08, 0.40, 0.40, 0.24])
    ax1.imshow(xenc.numpy(), cmap="cividis", aspect="auto")
    ax1.set_title("Input: one-hot encoding\nof '.emma.' transitions",
                  fontsize=9.5, color=INK)
    ax1.set_xlabel("character index (0–26)", fontsize=8)
    ax1.set_ylabel("example #", fontsize=8)
    ax1.tick_params(labelsize=7)

    # output probability matrix
    ax2 = fig.add_axes([0.55, 0.40, 0.40, 0.24])
    im = ax2.imshow(probs.detach().numpy(), cmap="magma", aspect="auto")
    ax2.set_title("Output: softmax probabilities\n(before any training)",
                  fontsize=9.5, color=INK)
    ax2.set_xlabel("next-char index", fontsize=8)
    ax2.set_ylabel("example #", fontsize=8)
    ax2.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # weight matrix
    ax3 = fig.add_axes([0.08, 0.10, 0.40, 0.22])
    im3 = ax3.imshow(W.numpy(), cmap="coolwarm", aspect="auto")
    ax3.set_title("Randomly initialised weights W (27×27)",
                  fontsize=9.5, color=INK)
    ax3.tick_params(labelsize=7)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # loss callout
    lax = fig.add_axes([0.55, 0.10, 0.40, 0.22])
    lax.axis("off")
    lax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor="#fdf3ef", edgecolor=ACCENT2,
                                 linewidth=1.4, transform=lax.transAxes))
    lax.text(0.5, 0.80, "Initial loss (untrained)", ha="center", fontsize=10,
             color=ACCENT2, fontweight="bold", transform=lax.transAxes)
    lax.text(0.5, 0.50, f"{nn_loss:.4f}", ha="center", fontsize=30,
             fontweight="bold", color=INK, transform=lax.transAxes)
    lax.text(0.5, 0.18,
             "Gradient descent on W will\ndrive this down toward the\nstatistical "
             "model's NLL.", ha="center", fontsize=8.5, color=MUTED,
             transform=lax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def page_summary(pdf, stats):
    fig = new_page()
    header_band(fig, "SECTION 06", "Takeaways & Next Steps", 7)

    items = [
        ("Two routes, one answer",
         "Counting bigrams and training a 27×27 linear layer arrive at the "
         "same probability table — one by direct statistics, one by gradient "
         "descent. Seeing this equivalence is the core lesson."),
        ("Smoothing matters",
         "Adding +1 to every count guarantees non-zero probabilities, so the "
         "negative log-likelihood never blows up on unseen transitions."),
        ("NLL is the compass",
         f"The fully-counted model reaches an average NLL of {stats['nll']:.4f} "
         "across the dataset — the baseline any learned model is measured against."),
        ("Bigram is the ceiling",
         "Looking only one character back limits quality. The natural next step "
         "is to widen the context window (trigrams, MLPs, and eventually a "
         "Transformer) — the trajectory of the makemore series."),
    ]
    y = 0.80
    for i, (title, body) in enumerate(items):
        a = panel(fig, [0.06, y - 0.155, 0.88, 0.145],
                  facecolor=LIGHT if i % 2 == 0 else "#fdf3ef")
        a.text(0.03, 0.74, f"{i+1}.  {title}", fontsize=12.5, fontweight="bold",
               color=INK, transform=a.transAxes)
        a.text(0.03, 0.34, body, fontsize=10, color=MUTED, va="center",
               transform=a.transAxes, wrap=True, linespacing=1.4)
        y -= 0.175

    fig.text(0.06, 0.075,
             "Reference: Andrej Karpathy — \"The spelled-out intro to language "
             "modeling: building makemore\".",
             fontsize=9, color=MUTED, style="italic")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    words = load_words()
    stoi, itos, N, P = build_model(words)
    samples = sample_names(P, itos, count=8)
    nll = dataset_nll(words, stoi, P)
    xs, ys, xenc, W, probs, nn_loss = neural_net_demo(words, stoi)

    stats = {
        "n_words": len(words),
        "n_bigrams": int(N.sum().item()),
        "nll": nll,
    }

    out_dir = Path(__file__).parent / "report"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "BigramModels_Report.pdf"

    with PdfPages(out_path) as pdf:
        page_cover(pdf, stats)
        page_overview(pdf, stats, words[:10])
        page_heatmap(pdf, N, itos)
        page_distribution(pdf, N, P, itos)
        page_sampling_nll(pdf, samples, nll, 27 * 27)
        page_neuralnet(pdf, xenc, W, probs, nn_loss)
        page_summary(pdf, stats)
        d = pdf.infodict()
        d["Title"] = "Bigram Language Models — Study Report"
        d["Author"] = "alpergocen35"
        d["Subject"] = "Character-level bigram name generator (makemore)"

    print(f"Wrote {out_path}")
    print(f"Stats: {stats}")
    print(f"Samples: {samples}")
    print(f"Neural-net initial loss: {nn_loss:.4f}")


if __name__ == "__main__":
    main()
