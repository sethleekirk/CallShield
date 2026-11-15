"""Interactive semantic scam-example search; run after build_semantic_index.py.
Run: python test_semantic_search.py --index semantic_index
"""
import argparse,json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',default='semantic_index');ap.add_argument('--k',type=int,default=20);args=ap.parse_args()
    folder=Path(args.index)
    emb=np.load(folder/'embeddings.npy',mmap_mode='r')
    with open(folder/'examples.jsonl',encoding='utf-8') as f: rows=[json.loads(s) for s in f]
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('Enter a call turn at a time; /new resets, /quit exits.')
    turns=[]
    while True:
        try: s=input('Call > ').strip()
        except (KeyboardInterrupt,EOFError):break
        if s=='/quit':break
        if s=='/new':turns=[];print('New call.');continue
        if not s:continue
        turns.append(s)
        # The query is the latest up-to-six turns, not an arbitrarily long transcript.
        query='\n'.join(turns[-6:])
        v=model.encode([query],normalize_embeddings=True,convert_to_numpy=True)[0].astype('float32')
        scores=np.asarray(emb@v)
        # Fetch extra neighbors, then collapse segments from the same conversation.
        candidates=np.argpartition(scores,-min(len(scores),max(100,args.k*15)))[-min(len(scores),max(100,args.k*15)):]
        candidates=candidates[np.argsort(scores[candidates])[::-1]]
        selected=[];seen=set()
        for i in candidates:
            r=rows[int(i)]
            if r['conversation_uid'] in seen:continue
            seen.add(r['conversation_uid']);selected.append((float(scores[i]),r))
            if len(selected)>=args.k:break
        scams=sum(r['label']=='YES' for _,r in selected)
        pct=100*scams/len(selected)
        # Similarity is a ranking metric, not a probability. Nonnegative weights are heuristic.
        weights=[max(0.0,sim)**4 for sim,_ in selected]
        weighted=100*sum(w for w,(_,r) in zip(weights,selected) if r['label']=='YES')/sum(weights) if sum(weights) else 0
        print(f'Nearest {len(selected)} unique conversations: {scams} scam, {len(selected)-scams} legitimate | Scam fraction: {pct:.1f}% | Similarity-weighted: {weighted:.1f}%')
        for sim,r in selected[:5]:
            snippet=r['text'].replace('\n',' / ')[:220]
            print(f"  {r['label']:3} | similarity {sim:.3f} | {r['domain']} | {snippet}")
        print('These percentages describe retrieved training examples, NOT the chance this call is a scam.\n')
if __name__=='__main__':main()
