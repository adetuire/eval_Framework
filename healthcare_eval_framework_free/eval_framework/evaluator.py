import csv, json, uuid, pathlib, yaml
from .metrics import accuracy as ACC
from .metrics import trustworthiness as TRUST
from .metrics import empathy as EMP
from .metrics import performance as PERF

def _model(prompt:str)->str:
    return "Placeholder answer."

def run_eval(cfg_path:str):
    cfg = yaml.safe_load(open(cfg_path))
    rows = list(csv.DictReader(open(cfg['datasets'][0]['path'])))
    prompts = [r['prompt'] for r in rows]
    refs = [r['reference'] for r in rows]
    preds = [_model(p) for p in prompts]

    results = {
        'accuracy': ACC.intrinsic_scores(refs, preds, cfg['metrics']['accuracy']['intrinsic']),
        'trustworthiness': {
            'safety': TRUST.safety_score(preds),
            'bias': TRUST.bias_score(preds)
        },
        'empathy': {
            'emotional_support': EMP.emotional_support_score(prompts, preds)
        },
        'performance': {
            'latency': PERF.measure_latency(_model, prompts[0]),
            'memory_mb': PERF.measure_memory(_model, prompts[0])
        }
    }
    total = 0.0
    for cat, weight in cfg['weights'].items():
        total += weight * (sum(results[cat].values())/len(results[cat]))
    results['total_score'] = total
    out = pathlib.Path(f"results_{uuid.uuid4().hex[:6]}.json")
    out.write_text(json.dumps(results, indent=2))
    print("Results saved to", out)
