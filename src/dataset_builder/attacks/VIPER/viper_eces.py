import random
from pathlib import Path

# VIPER resource directory (this file's directory)
_VIPER_DIR = Path(__file__).resolve().parent

random.seed(1)

def readD(fn: str):
    """
    Read mapping file (e.g., selected.neighbors).
    IMPORTANT: resolve relative paths against this file's directory,
    not the current working directory.
    """
    fn_path = Path(fn)
    if not fn_path.is_absolute():
        fn_path = (_VIPER_DIR / fn_path).resolve()

    h = {}
    with open(fn_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x = line.split()
            if len(x) < 2:
                continue
            a, b = x[0].strip(), x[1].strip()
            h[a] = b
    return h

def eces(prob, text: str):
    words = []
    h = readD("selected.neighbors")

    # keep original behavior
    truth = text  # noqa: F841

    text = (text or "").split()
    for line in text:
        if True:
            word = line
            ww = []
            p = random.random()
            if p > prob:
                words.append((word, word))
            else:
                max_try = 10
                while max_try:
                    w_idx = random.randint(0, len(word) - 1)
                    if word[w_idx].isalpha():
                        break
                    max_try -= 1
                if max_try == 0:
                    continue

                for wi, w in enumerate(word):
                    if wi == w_idx:
                        d = h.get(w, w)
                    else:
                        d = w
                    ww.append((d, w))

                words.append(("".join([c[0] for c in ww]), "".join([c[-1] for c in ww])))

    disturbed = " ".join([w[0] for w in words])
    return disturbed
