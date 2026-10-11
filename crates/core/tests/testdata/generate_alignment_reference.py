"""Synthetic score replays using the retained Python supplied-text policy and CTC 1.7.4."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for name in ('source', 'tokens', 'output'):
    p.add_argument(name, type=Path)
args = p.parse_args()
sys.path.insert(0, str(args.source))
from word_alignment import parse_vocabulary, prepare_text, align_ctc_scores, FrameClock, AcceptancePolicy
from alignment_model import CALIBRATION_ID, TOKEN_SHA256
from alignment_text import FUNCTIONAL_POLICY_ID
assert hashlib.sha256(args.tokens.read_bytes()).hexdigest() == TOKEN_SHA256
vocabulary = parse_vocabulary(args.tokens.read_text())
rows = []
for text, language in [('Go 12.5 e\u0301', 'en'), ('oo', 'en'), ('سَلَام', 'ar'), ('コーヒー', 'ja'), ('İ', 'en'), ('!!', 'en'), ('ش\u0651\u064e', 'ar'), ('a\u0315\u0300', 'en')]:
    prepared = prepare_text(text, vocabulary, language=language)
    frames = max(30, 3*len(prepared.targets)+8)
    scores = np.full((frames, len(vocabulary)), -30, dtype=np.float32)
    scores[:, 0] = 0
    for i, target in enumerate(prepared.targets):
        scores[3*i+4, :] = -30
        scores[3*i+4, target.token_id] = 0
    scores -= np.logaddexp.reduce(scores, axis=1, keepdims=True)
    calibration = CALIBRATION_ID if language == 'en' else FUNCTIONAL_POLICY_ID
    samples = 400 + (frames-1)*320
    result = align_ctc_scores(text, scores, vocabulary, clock=FrameClock(320,0,calibration), policy=AcceptancePolicy(-20,calibration), audio_start_sample=16000, audio_num_samples=samples, language=language)
    ids = list(dict.fromkeys([t.token_id for t in reversed(prepared.targets)] + [0]))
    rows.append(dict(text=text, language=language, samples=samples, token_ids=ids, tokens=[vocabulary[i] for i in ids], gathered=scores[:,ids].ravel().tolist(), expected=result))
args.output.write_text(json.dumps(dict(source_sha256={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in ('word_alignment.py','alignment_text.py')}, cases=rows),ensure_ascii=False,indent=2)+'\n')
print([(r['text'],r['expected']['status'],r['expected']['reason']) for r in rows])
