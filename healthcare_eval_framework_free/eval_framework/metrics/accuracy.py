from datasets import load_metric
def intrinsic_scores(refs, preds, names):
    out = {}
    for m in names:
        metric = load_metric(m)
        if m == 'bertscore':
            res = metric.compute(predictions=preds, references=refs, lang='en')
            out['bertscore_f1'] = sum(res['f1'])/len(res['f1'])
        else:
            res = metric.compute(predictions=preds, references=refs)
            key = list(res.keys())[0]
            score = res[key].mid.fmeasure if hasattr(res[key], 'mid') else res[key]
            out[m] = score
    return out
