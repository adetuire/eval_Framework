import time, psutil, os, gc
def measure_latency(fn, prompt, rep=3):
    times=[]
    for _ in range(rep):
        s=time.time();fn(prompt);times.append(time.time()-s)
    return sum(times)/len(times)
def measure_memory(fn, prompt):
    proc=psutil.Process(os.getpid())
    before=proc.memory_info().rss/(1024**2)
    fn(prompt);gc.collect()
    after=proc.memory_info().rss/(1024**2)
    return max(after, before)
