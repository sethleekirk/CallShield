"""Balanced semantic retrieval: top K unique scam and legitimate conversations.
Run from any folder: python test_semantic_search_balanced.py
Optional: --index PATH --k 20
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def top_unique(scores, indices, rows, k):
    """Highest-scoring segment per conversation, sorted descending."""
    # Search all rows of this label so even uncommon conversations can qualify.
    ordered = indices[np.argsort(scores[indices])[::-1]]
    selected = []
    seen = set()
    for idx in ordered:
        row = rows[int(idx)]
        uid = row.get('conversation_uid', int(idx))
        if uid in seen:
            continue
        seen.add(uid)
        selected.append((float(scores[idx]), row))
        if len(selected) == k:
            break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='semantic_index')
    parser.add_argument('--k', type=int, default=20, help='Examples PER label')
    args = parser.parse_args()
    if args.k < 1:
        parser.error('--k must be at least 1')

    script_dir = Path(__file__).resolve().parent
    folder = Path(args.index)
    if not folder.is_absolute():
        folder = script_dir / folder
    emb = np.load(folder / 'embeddings.npy', mmap_mode='r')
    with (folder / 'examples.jsonl').open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != len(emb):
        raise ValueError(f'Index mismatch: {len(emb)} embeddings, {len(rows)} examples')

    scam_indices = np.array([i for i, row in enumerate(rows) if row['label'] == 'YES'], dtype=np.int64)
    legit_indices = np.array([i for i, row in enumerate(rows) if row['label'] == 'NO'], dtype=np.int64)
    if not len(scam_indices) or not len(legit_indices):
        raise ValueError('Both YES and NO examples are required')

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f'Loaded {len(rows):,} indexed segments. Comparing top {args.k} UNIQUE conversations per label.')
    print('Enter a call turn at a time; /new resets, /quit exits.')
    turns = []
    while True:
        try:
            s = input('Call > ').strip()
        except (KeyboardInterrupt, EOFError):
            break
        if s == '/quit':
            break
        if s == '/new':
            turns = []
            print('New call.')
            continue
        if not s:
            continue
        turns.append(s)
        query = '\n'.join(turns[-6:])
        v = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype('float32')
        scores = np.asarray(emb @ v)

        scams = top_unique(scores, scam_indices, rows, args.k)
        legit = top_unique(scores, legit_indices, rows, args.k)

        # Fourth-power similarity weights prioritize very close matches.
        # This is a heuristic score, NOT a calibrated probability.
        scam_weight = sum(max(0.0, sim) ** 4 for sim, _ in scams)
        legit_weight = sum(max(0.0, sim) ** 4 for sim, _ in legit)
        total_weight = scam_weight + legit_weight
        weighted_pct = 100 * scam_weight / total_weight if total_weight else 50.0
        avg_scam = sum(sim for sim, _ in scams) / len(scams)
        avg_legit = sum(sim for sim, _ in legit) / len(legit)
        print(f'\nTop {len(scams)} scam vs top {len(legit)} legitimate conversations')
        print(f'Average similarity: scam {avg_scam:.3f} | legitimate {avg_legit:.3f} | advantage {avg_scam-avg_legit:+.3f}')
        print(f'Similarity-weighted comparison: scam {weighted_pct:.1f}% | legitimate {100-weighted_pct:.1f}%')
        print('Top 5 scam matches:')
        for sim, row in scams[:5]:
            snippet = row['text'].replace('\n', ' / ')[:180]
            print(f"  {sim:.3f} | {row.get('domain', '')} | {snippet}")
        print('Top 5 legitimate matches:')
        for sim, row in legit[:5]:
            snippet = row['text'].replace('\n', ' / ')[:180]
            print(f"  {sim:.3f} | {row.get('domain', '')} | {snippet}")
        print('Note: This is a relative similarity score, NOT a scam probability.\n')


if __name__ == '__main__':
    main()
