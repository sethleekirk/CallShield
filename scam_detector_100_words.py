"""Classify a completed conversation using balanced MiniLM retrieval; explain YES with local Ollama Qwen.

Install: pip install numpy sentence-transformers
Run: python scam_detector_with_qwen.py --model qwen3.5:2b
Paste one conversation turn per line, then type /analyze. /quit exits.
Alternatively: python scam_detector_with_qwen.py --file conversation.txt

Uses 20-word groups across the FULL conversation, with 20 unique
neighbors per label and similarity**4 weighting per group. Group scores
are averaged. This differs from the original last-six-turn evaluator.
"""
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def nearest_unique(scores, candidates, uid_codes, k):
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
            if uid not in seen:
                seen.add(uid)
                chosen.append(float(scores[idx]))
                if len(chosen) >= k:
                    return chosen
        if pool == n:
            return chosen
        pool = min(n, pool * 2)


def full_conversation_windows(transcript, tokenizer, words_per_group=20, context_words=5):
    """Non-overlapping 20-word cores, with up to 5 words of context on each side."""
    words = transcript.split()
    windows = []
    for start in range(0, len(words), words_per_group):
        end = min(start + words_per_group, len(words))
        group = ' '.join(words[max(0, start - context_words):min(len(words), end + context_words)])
        token_ids = tokenizer(group, add_special_tokens=False, truncation=False)['input_ids']
        for offset in range(0, len(token_ids), 254):
            piece = token_ids[offset:offset + 254]
            windows.append(tokenizer.decode(piece, skip_special_tokens=True))
    return windows


def classify_call(highest_pct, average_pct):
    """Classification based on tested peak and mean section scores (0-100)."""
    if highest_pct >= 60 or average_pct >= 52:
        return "YES"
    if average_pct >= 50:
        return "YES?"
    if average_pct >= 48:
        return "NO?"
    return "NO"

def explain_with_qwen(transcript, model, url, timeout):
    system = """
You are a phone scam warning explanation assistant.

A separate detection system has already flagged this
conversation as potentially suspicious.

Your task is to examine the FULL conversation and identify
specific warning signs.

Look for:
- Impersonation of trusted organizations
- Unsolicited security or account warnings
- Requests for remote access
- Requests for money or sensitive information
- Pressure to act immediately
- Discouraging independent verification
- Threats of negative consequences for refusing

Explain the strongest 2-4 warning signs using evidence
from the actual conversation.

Do not assume the caller is legitimate merely because
they claim to represent a trusted organization.

Do not invent evidence or claim fraud is proven.

If there are no meaningful warning signs, say so.

Before explaining a warning sign, check which speaker said it.
Never attribute the customer's statements to the caller.
Only describe behavior supported by the transcript, and quote accurately.
If the evidence is ambiguous, acknowledge the uncertainty.

Keep your response under 120 words.
"""

    payload = {
    'model': model,
    'stream': False,
    'think': False,
    'messages': [
        {'role': 'system', 'content': system},
        {
            'role': 'user',
            'content': (
                'Analyze this complete conversation:\n'
                '<transcript>\n'
                + transcript +
                '\n</transcript>'
            )
        },
    ],
    'options': {
        'temperature': 0.1,
        'num_predict': 500,
    },
}
    req = urllib.request.Request(url.rstrip('/') + '/api/chat',
                                 data=json.dumps(payload).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    return result.get('message', {}).get('content', '').strip() or '(Qwen returned no explanation.)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', default='semantic_index')
    ap.add_argument('--file', help='Text file containing a complete conversation')
    ap.add_argument('--model', default='qwen3.5:2b', help='Installed Ollama model name')
    ap.add_argument('--ollama', default='http://127.0.0.1:11434')
    ap.add_argument('--timeout', type=int, default=180)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--group-words', type=int, default=20, help='Words per MiniLM group (default: 20)')
    args = ap.parse_args()
    if args.k < 1 or args.group_words < 1:
        ap.error('--k and --group-words must be positive')
    base = Path(__file__).resolve().parent
    index = Path(args.index)
    if not index.is_absolute():
        index = base / index
    print('Loading semantic index and MiniLM...')
    emb = np.asarray(np.load(index / 'embeddings.npy', mmap_mode='r'), dtype=np.float32)
    with (index / 'examples.jsonl').open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != len(emb):
        raise ValueError('Index embeddings and examples have different lengths')
    labels = np.array([r['label'] for r in rows])
    candidates = {label: np.flatnonzero(labels == label) for label in ('YES', 'NO')}
    if any(not len(v) for v in candidates.values()):
        raise ValueError('Index must contain both YES and NO examples')
    uid_map = {uid: i for i, uid in enumerate(set(str(r['conversation_uid']) for r in rows))}
    uid_codes = np.fromiter((uid_map[str(r['conversation_uid'])] for r in rows), dtype=np.int32)
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f'Ready: {len(rows):,} segments. Qwen model: {args.model}')

    def analyze(transcript):
        turns = [line.strip() for line in transcript.splitlines() if line.strip()]
        if not turns:
            print('No conversation supplied.'); return
        # Score every part of the transcript, not only the final six turns.
        windows = full_conversation_windows('\n'.join(turns), encoder.tokenizer,
                                            args.group_words)
        vectors = encoder.encode(windows, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)
        percentages = []
        for v in vectors:
            scores = emb @ v
            scam = nearest_unique(scores, candidates['YES'], uid_codes, args.k)
            legit = nearest_unique(scores, candidates['NO'], uid_codes, args.k)
            sw = sum(max(0., x)**4 for x in scam)
            lw = sum(max(0., x)**4 for x in legit)
            percentages.append(100 * sw / (sw + lw) if sw + lw else 50.)
        pct = float(np.mean(percentages))
        peak = max(percentages)
        result = classify_call(peak, pct)
        flagged = result in ('YES', 'YES?')
        print(f'Analyzed entire conversation in {len(windows)} MiniLM segment(s).')
        print(f'\nResult: {result} | Highest section: {peak:.1f}%')
        print(f'Scam similarity: {pct:.1f}% | Legitimate similarity: {100-pct:.1f}%')
        print('This is a relative similarity heuristic, not a calibrated probability.')
        if not flagged:
            print('Qwen skipped (NO). A NO result does not guarantee safety.\n')
            return
        print('\nQwen explanation (analyzing the FULL transcript):')
        try:
            explanation = explain_with_qwen('\n'.join(turns), args.model, args.ollama, args.timeout)
            print(explanation + '\n')
        except urllib.error.URLError as e:
            print(f'Could not reach Ollama: {e}. Start Ollama and check the model name.\n')
        except (TimeoutError, ValueError) as e:
            print(f'Qwen explanation failed: {e}\n')

    if args.file:
        file = Path(args.file)
        if not file.is_absolute():
            file = base / file
        analyze(file.read_text(encoding='utf-8'))
    else:
        print('Paste the COMPLETE conversation, one turn per line. Type /analyze to classify; /quit to exit.')
        turns = []
        while True:
            try:
                line = input('Call > ')
            except (EOFError, KeyboardInterrupt):
                print(); break
            if line.strip().lower() == '/quit':
                break
            if line.strip().lower() == '/analyze':
                analyze('\n'.join(turns))
                turns.clear()
                continue
            if line.strip().startswith('/'):
                print('Unknown command. Use /analyze or /quit.')
                continue
            if line.strip():
                turns.append(line.strip())


if __name__ == '__main__':
    main()
