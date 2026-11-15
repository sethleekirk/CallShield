"""Batch-evaluate the existing balanced MiniLM retrieval heuristic.

Examples:
  python evaluate_semantic_detector.py --input validation.jsonl --limit-per-label 50
  python evaluate_semantic_detector.py --input train.jsonl --limit-per-label 20

The second command is exploratory ONLY: training data is not a held-out evaluation.
Requires: numpy, sentence-transformers. Does not modify the index.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Bad JSON at {path}:{line_no}: {exc}') from exc


def nearest_unique(scores, candidates, uid_codes, k, excluded_code):
    """Find highest-scoring segment from each of k distinct conversations."""
    # An expanding candidate pool avoids sorting every row for ordinary cases.
    n = len(candidates)
    pool = min(n, max(500, 20 * k))
    while True:
        vals = scores[candidates]
        if pool < n:
            selected = np.argpartition(vals, -pool)[-pool:]
        else:
            selected = np.arange(n)
        selected = selected[np.argsort(vals[selected])[::-1]]
        out = []
        seen = set()
        for local_idx in selected:
            idx = int(candidates[local_idx])
            uid = int(uid_codes[idx])
            if uid == excluded_code or uid in seen:
                continue
            seen.add(uid)
            out.append(float(scores[idx]))
            if len(out) == k:
                return out
        if pool == n:
            return out
        pool = min(n, pool * 2)


def summarize(scores):
    a = np.asarray(scores, dtype=float)
    diffs = np.diff(a)
    return {
        'first_score': round(float(a[0]), 3),
        'final_score': round(float(a[-1]), 3),
        'mean_score': round(float(a.mean()), 3),
        'min_score': round(float(a.min()), 3),
        'max_score': round(float(a.max()), 3),
        'score_std': round(float(a.std()), 3),
        'change_std': round(float(diffs.std()), 3) if len(diffs) else 0.0,
        'mean_abs_change': round(float(np.abs(diffs).mean()), 3) if len(diffs) else 0.0,
        'max_abs_change': round(float(np.abs(diffs).max()), 3) if len(diffs) else 0.0,
        'net_change': round(float(a[-1] - a[0]), 3),
        'pct_turns_above_50': round(float((a > 50).mean() * 100), 3),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='validation.jsonl', help='Held-out JSONL with conversation_uid, text, label YES/NO')
    ap.add_argument('--index', default='semantic_index')
    ap.add_argument('--output', default='evaluation_results')
    ap.add_argument('--limit-per-label', type=int, default=50, help='Maximum conversations from each label; 0 means all')
    ap.add_argument('--k', type=int, default=20, help='Unique neighbors per class')
    ap.add_argument('--window', type=int, default=6, help='Most recent turns in each query')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.k < 1 or args.window < 1 or args.limit_per_label < 0 or args.batch_size < 1:
        ap.error('k, window, batch-size must be positive; limit-per-label must be nonnegative')

    base = Path(__file__).resolve().parent
    def resolve(s):
        p = Path(s)
        return p if p.is_absolute() else base / p

    input_path, index_dir, output_dir = map(resolve, (args.input, args.index, args.output))
    if not input_path.exists():
        ap.error(f'Cannot find {input_path}. Supply --input PATH to your validation/test JSONL.')
    output_dir.mkdir(parents=True, exist_ok=True)
    emb = np.load(index_dir / 'embeddings.npy', mmap_mode='r')
    rows = list(read_jsonl(index_dir / 'examples.jsonl'))
    if len(emb) != len(rows):
        raise ValueError('Index rows do not match embedding count')
    labels = np.array([r['label'] for r in rows])
    candidates = {label: np.flatnonzero(labels == label) for label in ('YES', 'NO')}
    if any(len(v) == 0 for v in candidates.values()):
        raise ValueError('Index needs both YES and NO rows')
    uid_map = {uid: i for i, uid in enumerate(set(str(r['conversation_uid']) for r in rows))}
    uid_codes = np.fromiter((uid_map[str(r['conversation_uid'])] for r in rows), dtype=np.int32)

    grouped = defaultdict(list)
    for item in read_jsonl(input_path):
        if item.get('label') in ('YES', 'NO') and item.get('text'):
            uid = str(item.get('conversation_uid', ''))
            if not uid:
                raise ValueError('Each test conversation needs a conversation_uid')
            grouped[item['label']].append(item)
    rng = np.random.default_rng(args.seed)
    chosen = []
    for label in ('YES', 'NO'):
        # Avoid duplicate UIDs in input, even if a file contains repeated rows.
        unique = {str(item['conversation_uid']): item for item in grouped[label]}
        pool = list(unique.values())
        rng.shuffle(pool)
        chosen.extend(pool[:args.limit_per_label or None])
    if not chosen:
        raise ValueError('No labeled conversations found in input')
    overlap = sum(str(item['conversation_uid']) in uid_map for item in chosen)
    print(f'Index: {len(rows):,} segments. Evaluating {len(chosen)} conversations ({overlap} UIDs in index).', flush=True)
    if overlap:
        print('WARNING: Some test conversations are in the index. Self-matches will be excluded, but this is NOT a held-out evaluation.', flush=True)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb = np.asarray(emb, dtype=np.float32)
    per_conversation, per_turn = [], []
    confusion = {'YES': {'YES': 0, 'NO': 0}, 'NO': {'YES': 0, 'NO': 0}}
    for number, item in enumerate(chosen, 1):
        uid = str(item['conversation_uid'])
        true = item['label']
        turns = [line.strip() for line in item['text'].splitlines() if line.strip()]
        if not turns:
            continue
        queries = ['\n'.join(turns[max(0, i - args.window + 1):i + 1]) for i in range(len(turns))]
        vectors = model.encode(queries, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        all_scores = []
        for start in range(0, len(vectors), args.batch_size):
            sims_batch = emb @ vectors[start:start + args.batch_size].T
            for j in range(sims_batch.shape[1]):
                sims = sims_batch[:, j]
                scam = nearest_unique(sims, candidates['YES'], uid_codes, args.k, uid_map.get(uid, -1))
                legit = nearest_unique(sims, candidates['NO'], uid_codes, args.k, uid_map.get(uid, -1))
                if not scam or not legit:
                    raise ValueError('Not enough examples after excluding test conversation')
                sw = sum(max(0., x) ** 4 for x in scam)
                lw = sum(max(0., x) ** 4 for x in legit)
                pct = 100 * sw / (sw + lw) if sw + lw else 50.
                turn_no = start + j + 1
                all_scores.append(pct)
                per_turn.append({'conversation_uid': uid, 'true_label': true, 'domain': item.get('domain', ''),
                                 'turn': turn_no, 'turn_text': turns[turn_no - 1],
                                 'scam_similarity_pct': round(pct, 3),
                                 'delta_from_previous': round(pct - all_scores[-2], 3) if len(all_scores) > 1 else 0.,
                                 'scam_avg_similarity': round(float(np.mean(scam)), 4),
                                 'legit_avg_similarity': round(float(np.mean(legit)), 4)})
        final_pred = 'YES' if all_scores[-1] > 50 else 'NO'  # ties -> NO
        confusion[true][final_pred] += 1
        per_conversation.append({'conversation_uid': uid, 'true_label': true, 'predicted_label': final_pred,
                                 'correct': int(true == final_pred), 'domain': item.get('domain', ''),
                                 'turn_count': len(turns), **summarize(all_scores)})
        print(f'[{number}/{len(chosen)}] {true} -> {final_pred} | final={all_scores[-1]:.1f}% | {len(turns)} turns', flush=True)

    write_csv(output_dir / 'conversations.csv', per_conversation)
    write_csv(output_dir / 'turns.csv', per_turn)
    tp, fn = confusion['YES']['YES'], confusion['YES']['NO']
    fp, tn = confusion['NO']['YES'], confusion['NO']['NO']
    safe_div = lambda a, b: a / b if b else None
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    metrics = {
        'notes': 'Scores are relative similarity heuristics, not calibrated scam probabilities. Threshold 50% is exploratory.',
        'input': str(input_path), 'index': str(index_dir), 'overlap_uids': overlap,
        'evaluated_conversations': len(per_conversation), 'settings': vars(args),
        'confusion': {'true_scam_pred_scam': tp, 'true_scam_pred_legit': fn,
                      'true_legit_pred_scam': fp, 'true_legit_pred_legit': tn},
        'accuracy': safe_div(tp + tn, tp + tn + fp + fn),
        'scam_precision': precision, 'scam_recall': recall,
        'scam_f1': safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None,
        'legit_specificity': safe_div(tn, tn + fp),
        'by_label': {
            label: {key: round(float(np.mean([r[key] for r in per_conversation if r['true_label'] == label])), 3)
                    for key in ('final_score', 'mean_score', 'score_std', 'change_std', 'mean_abs_change', 'max_abs_change', 'net_change')}
            for label in ('YES', 'NO') if any(r['true_label'] == label for r in per_conversation)
        }
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print('\nRESULTS')
    print(json.dumps({k: metrics[k] for k in ('evaluated_conversations', 'overlap_uids', 'confusion', 'accuracy', 'scam_precision', 'scam_recall', 'scam_f1', 'by_label')}, indent=2))
    print(f'\nSaved: {output_dir / "summary.json"}, conversations.csv, turns.csv')


if __name__ == '__main__':
    main()
