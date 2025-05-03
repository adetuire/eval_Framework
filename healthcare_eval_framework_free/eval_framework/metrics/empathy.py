from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np
MODEL = 'nateraw/bert-base-uncased-emotion'
_tok = AutoTokenizer.from_pretrained(MODEL)
_mod = AutoModelForSequenceClassification.from_pretrained(MODEL)
_mod.eval()
SUPPORT_IDX = [2,3,5]  # joy, love, surprise
@torch.inference_mode()
def emotional_support_score(prompts, responses):
    texts = [f"{p} [SEP] {r}" for p,r in zip(prompts, responses)]
    inputs = _tok(texts, padding=True, truncation=True, return_tensors='pt')
    logits = _mod(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    supportive = probs[:, SUPPORT_IDX].sum(axis=1)
    scores = supportive * 4.0
    return float(scores.mean())
