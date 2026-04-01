from datasets import load_dataset
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("Loading dataset in streaming mode to extract column names rapidly...")
try:
    ds = load_dataset('ministere-culture/comparia-votes', split='train', streaming=True)
    first_row = next(iter(ds))
    print("COLUMNS FROM HF:", list(first_row.keys()))
except Exception as e:
    print("Failed HF:", e)

try:
    ds_react = load_dataset('ministere-culture/comparia-reactions', split='train', streaming=True)
    first_row_react = next(iter(ds_react))
    print("COLUMNS FROM HF REACTIONS:", list(first_row_react.keys()))
except Exception as e:
    print("Failed HF Reactions:", e)
