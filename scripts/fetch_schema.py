import urllib.request
import pyarrow.parquet as pq

url = 'https://object.data.gouv.fr/ministere-culture/COMPARIA/votes.parquet'
req = urllib.request.Request(url, method='HEAD')
try:
    with urllib.request.urlopen(req) as resp:
        size = int(resp.headers['Content-Length'])
        print(f"File size: {size / 1024 / 1024:.2f} MB")
except Exception as e:
    print("Cannot get HEAD", e)

print("Reading schema using pyarrow without downloading whole file...")
# fastparquet or pyarrow over fsspec can read schema efficiently, but pandas over simple HTTP URL downloads entirely.
try:
    import pandas as pd
    # download a 100kb chunk? No, just download it to disk. 
    import os
    if not os.path.exists('votes_sample.parquet'):
        print("Downloading votes.parquet entirely to disk (will take a minute)...")
        urllib.request.urlretrieve(url, 'votes.parquet')
    
    df = pd.read_parquet('votes.parquet')
    print("COLUMNS IN VOTES:", df.columns.tolist())
except Exception as e:
    print("Error:", e)
