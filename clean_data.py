# clean_corpus_enchant_scitech.py
import re
import enchant
from difflib import get_close_matches
from tqdm import tqdm

# ---------------- CONFIG ----------------
input_file = "Data/test.txt"          # your input file
output_file = "cleaned_test.txt" # output file
lang_dict = "en_US"
# ----------------------------------------

# Initialize dictionary
d = enchant.Dict(lang_dict)

# Whitelist of technical/scientific/CS roots
sci_tech_roots = [
    # General science
    "bio", "neuro", "physio", "pharma", "chem", "astro", "thermo", "micro",
    "photo", "psycho", "cyto", "gen", "path", "graph", "meter", "scope",
    "electro", "socio", "chrono", "geo", "hydro", "cardio", "derma",
    "spectro", "morph", "bot", "zoo", "philo", "meta", "macro", "nano",
    "proto", "poly", "oxy", "chloro", "hydro", "therm", "radi", "organo",
    "enzyme", "acid", "cell", "vaccine", "protein", "virus", "bacter", "atom",
    "optic", "ionic", "neural", "chrom", "physic", "quant", "catal", "plas",
    "phon", "spectr", "phyt", "photo", "hydrogen", "carbon", "nucle", "ionic",
    "magnet", "electro", "chemic", "optic", "therap", "clinic", "cognit",
    "astro", "planet", "lith", "geo", "paleo", "ethno", "analy", "experiment",
    "bioinfo", "proteo", "neurotrans", "immuno",

    # Computer science / math / AI
    "algo", "data", "logic", "digit", "comput", "inform", "neuron", "ai",
    "intellig", "learn", "matrix", "vector", "tensor", "network", "graph",
    "model", "stat", "predict", "classif", "cluster", "analyt", "program",
    "bit", "byte", "crypto", "quantum", "boolean", "math", "calc", "linear",
    "algorith", "code", "machine", "deep", "embed", "transform", "token",
    "genetic", "eigen", "pca", "svd", "backprop", "grad", "sgd", "loss",
    "kernel", "regress", "infer", "inference", "probab", "stochast", "vector",
    "graphene", "topolog", "entropy", "gradient", "activation", "dataset",
    "dataset", "api", "function", "method", "variable"
]

def is_scitech(word: str) -> bool:
    """Check if word resembles a scientific/technical term."""
    for root in sci_tech_roots:
        if root in word:
            return True
    return False

# Load words
with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    words = [line.strip().lower() for line in f if line.strip()]

cleaned = []

print("Cleaning and filtering corpus...")

for word in tqdm(words):
    # Skip non-alphabetic
    if not re.fullmatch(r"[a-z]+", word):
        continue

    # Valid dictionary word
    if d.check(word):
        cleaned.append(word)
        continue

    # Valid sci/tech term
    if is_scitech(word):
        cleaned.append(word)
        continue

    # Try correcting simple typos
    suggestions = d.suggest(word)
    if suggestions:
        top = suggestions[0].lower()
        if len(get_close_matches(word, [top], n=1, cutoff=0.75)) > 0:
            cleaned.append(top)
            continue

# Deduplicate while preserving order
seen = set()
final_words = []
for w in cleaned:
    if w not in seen:
        seen.add(w)
        final_words.append(w)

# Save cleaned output
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(final_words))

print(f"\nOriginal words: {len(words)}")
print(f"Cleaned words: {len(final_words)}")
print(f"Retention: {len(final_words) / len(words) * 100:.2f}%")
print(f"Cleaned corpus saved as {output_file}")
