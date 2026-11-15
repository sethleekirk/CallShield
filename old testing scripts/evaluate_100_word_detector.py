"""Evaluate the exact 100-word MiniLM scoring logic (no Qwen calls).

Run: python evaluate_100_word_detector.py --input validation.jsonl --limit-per-label 50
Requires scam_detector_100_words.py alongside this script and semantic_index/.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from scam_detector_100_words import full_conversation_windows


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def nearest_unique(scores, candidates, uid_codes, k, excluded_code=-1):
    """Same expanding-pool, unique-conversation ranking as the detector.

    Excludes the evaluated UID if it is present in the index.
    """
    n = len(candidates)
    pool = min(n, max(500, 20 * k))
    while True:
        vals = scores[candidates]
        selected = np.argpartition(vals, -pool)[-pool:] if pool < n else np.arange(n)
        selected = selected[np.argsort(vals[selected])[::-1]]
        chosen, seen = [], set()
        for local in selected:
            idx = int(candidates[local])
            uid = int(uid_codes[idx])
            if uid == excluded_code or uid in seen:
                continue
            seen.add(uid)
            chosen.append(float(scores[idx]))
            if len(chosen) >= k:
                return chosen
        if pool == n:
            return chosen
        pool = min(n, pool * 2)


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def ratio(a, b):
    return a / b if b else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='validation.jsonl')
    ap.add_argument('--index', default='semantic_index')
    ap.add_argument('--output', default='evaluation_100_words')
    ap.add_argument('--limit-per-label', type=int, default=50, help='0 = all')
    ap.add_argument('--group-words', type=int, default=100)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.limit_per_label < 0 or min(args.group_words, args.k, args.batch_size) < 1:
        ap.error('Invalid argument: group-words, k and batch-size must be positive; limit >= 0')
    base = Path(__file__).resolve().parent
    def resolve(name):
        path = Path(name)
        return path if path.is_absolute() else base / path
    input_path, index_path, output_path = map(resolve, (args.input, args.index, args.output))
    if not input_path.is_file():
        ap.error(f'Input file not found: {input_path}')
    output_path.mkdir(parents=True, exist_ok=True)

    emb = np.asarray(np.load(index_path / 'embeddings.npy', mmap_mode='r'), dtype=np.float32)
    index_rows = list(read_jsonl(index_path / 'examples.jsonl'))
    if len(index_rows) != len(emb):
        raise ValueError('Index embeddings and metadata lengths differ')
    labels = np.array([r['label'] for r in index_rows])
    candidates = {label: np.flatnonzero(labels == label) for label in ('YES', 'NO')}
    if any(len(c) == 0 for c in candidates.values()):
        raise ValueError('Index must contain YES and NO examples')
    uid_map = {uid: i for i, uid in enumerate(set(str(r['conversation_uid']) for r in index_rows))}
    uid_codes = np.fromiter((uid_map[str(r['conversation_uid'])] for r in index_rows), dtype=np.int32)

    grouped = defaultdict(dict)
    for item in read_jsonl(input_path):
        if item.get('label') in ('YES', 'NO') and item.get('text') and item.get('conversation_uid') is not None:
            grouped[item['label']][str(item['conversation_uid'])] = item
    rng = np.random.default_rng(args.seed)
    chosen = []
    for label in ('YES', 'NO'):
        pool = list(grouped[label].values())
        rng.shuffle(pool)
        chosen.extend(pool[:args.limit_per_label or None])
    if not chosen:
        raise ValueError('No valid labeled conversations in input')
    overlap = sum(str(item['conversation_uid']) in uid_map for item in chosen)
    print(f'Index: {len(index_rows):,} segments; testing {len(chosen)} conversations; UID overlap: {overlap}', flush=True)
    if overlap:
        print('WARNING: index overlap detected. Self-matches excluded, but not a held-out evaluation.', flush=True)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    results, segment_results = [], []
    confusion = {'YES': {'YES': 0, 'NO': 0}, 'NO': {'YES': 0, 'NO': 0}}
    for i, item in enumerate(chosen, 1):
        uid, truth = str(item['conversation_uid']), item['label']
        transcript = '\n'.join(line.strip() for line in item['text'].splitlines() if line.strip())
        windows = full_conversation_windows(transcript, model.tokenizer, args.group_words)
        if not windows:
            continue
        vectors = model.encode(windows, batch_size=args.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        percentages = []
        for start in range(0, len(vectors), args.batch_size):
            similarities = emb @ vectors[start:start + args.batch_size].T
            for j in range(similarities.shape[1]):
                sims = similarities[:, j]
                scam = nearest_unique(sims, candidates['YES'], uid_codes, args.k, uid_map.get(uid, -1))
                legit = nearest_unique(sims, candidates['NO'], uid_codes, args.k, uid_map.get(uid, -1))
                sw = sum(max(0., x) ** 4 for x in scam)
                lw = sum(max(0., x) ** 4 for x in legit)
                pct = 100 * sw / (sw + lw) if sw + lw else 50.
                percentages.append(pct)
                segment_results.append({'conversation_uid': uid, 'true_label': truth,
                                        'segment': start + j + 1, 'scam_similarity_pct': round(pct, 3)})
        score = float(np.mean(percentages))
        pred = 'YES' if score > 50 else 'NO'
        confusion[truth][pred] += 1
        results.append({'conversation_uid': uid, 'true_label': truth, 'predicted_label': pred,
                        'correct': int(pred == truth), 'scam_similarity_pct': round(score, 3),
                        'legitimate_similarity_pct': round(100 - score, 3),
                        'segments': len(windows), 'domain': item.get('domain', '')})
        print(f'[{i}/{len(chosen)}] {truth} -> {pred} | scam={score:.1f}% | {len(windows)} segments', flush=True)
    tp, fn = confusion['YES']['YES'], confusion['YES']['NO']
    fp, tn = confusion['NO']['YES'], confusion['NO']['NO']
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    summary = {'input': str(input_path), 'index': str(index_path), 'settings': vars(args),
               'evaluated_conversations': len(results), 'overlap_uids': overlap,
               'confusion': {'true_scam_pred_scam': tp, 'true_scam_pred_legit': fn,
                             'true_legit_pred_scam': fp, 'true_legit_pred_legit': tn},
               'accuracy': ratio(tp + tn, tp + fn + fp + tn),
               'scam_precision': precision, 'scam_recall': recall,
               'scam_f1': ratio(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
               'legit_specificity': ratio(tn, tn + fp),
               'notes': 'Full-conversation 100-word groups; per-group balanced top-k unique neighbors, similarity**4, mean group score, threshold >50. No Qwen. Not calibrated probabilities.'}
    write_csv(output_path / 'conversations.csv', results)
    write_csv(output_path / 'segments.csv', segment_results)
    (output_path / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nRESULTS\n' + json.dumps({k: summary[k] for k in ('evaluated_conversations', 'overlap_uids', 'confusion', 'accuracy', 'scam_precision', 'scam_recall', 'scam_f1')}, indent=2))
    print(f'\nSaved results in: {output_path}')


if __name__ == '__main__':
    main()
