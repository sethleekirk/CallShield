"""Build a local MiniLM nearest-neighbor index from labeled training conversations.
Install: pip install sentence-transformers numpy scikit-learn joblib
Run: python build_semantic_index.py --input train.jsonl
"""
import argparse, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib


def segments(text):
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    # Overlapping windows preserve context while avoiding long-text truncation.
    for start in range(0,len(lines),3):
        part='\n'.join(lines[start:start+6])
        if part:
            yield part


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='train.jsonl')
    ap.add_argument('--output',default='semantic_index')
    ap.add_argument('--batch-size',type=int,default=64)
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    file_path = Path(__file__).resolve().parent / "train.jsonl"

    with open(file_path, encoding='utf-8') as f:
        for line in f:
            item=json.loads(line)
            label=item['label']
            if label not in ('YES','NO'): continue
            for seg in segments(item['text']):
                rows.append({'text':seg,'label':label,'conversation_uid':item['conversation_uid'],'domain':item.get('domain','')})
    print(f'Encoding {len(rows):,} segments. First run can take a while on CPU.',flush=True)
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb=model.encode([r['text'] for r in rows],batch_size=args.batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype('float32')
    np.save(out/'embeddings.npy',emb)
    with open(out/'examples.jsonl','w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'Saved {len(rows):,} segments to {out.resolve()}')

if __name__=='__main__':main()
