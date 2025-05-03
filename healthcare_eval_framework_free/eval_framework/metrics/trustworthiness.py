try:
    from trustllm import evaluator as _eval
except ImportError:
    _eval = None

def _fallback():
    return 0.0

def safety_score(texts):
    if _eval:
        return _eval.evaluate(texts, aspects=['toxicity'])['toxicity']['score']
    return _fallback()

def bias_score(texts):
    if _eval:
        return _eval.evaluate(texts, aspects=['bias'])['bias']['score']
    return _fallback()
