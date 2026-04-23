#!/usr/bin/env python3
import requests
import os
import sys

url = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"
save_path = os.path.expanduser("~/.cellpose/models/cpsam")
proxy = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}

os.makedirs(os.path.dirname(save_path), exist_ok=True)
if os.path.exists(save_path):
    os.remove(save_path)

print(f"Downloading cpsam model (~1.1GB)...")

r = requests.get(url, stream=True, timeout=120, proxies=proxy, allow_redirects=True)
r.raise_for_status()
total = int(r.headers.get('Content-Length', 0))
print(f"Total size: {total/(1024*1024):.1f} MB")

downloaded = 0
with open(save_path, 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024*1024):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100 if total else 0
            print(f"  {downloaded/(1024*1024):.1f}/{total/(1024*1024):.1f} MB ({pct:.1f}%)")

print(f"Done! File size: {os.path.getsize(save_path)} bytes")
