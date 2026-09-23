"""Queue: train + evaluate every variant/seed with W single-threaded workers pinned to cores."""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

JOBS = [("scale", 0), ("untied", 0), ("nodelta", 0), ("scale", 1), ("local", 0), ("single", 0), ("scale", 2)]
CORES = [int(c) for c in os.environ.get("CORES", "1,2,3").split(",")]
EVAL = ["--Ls", "8,16,32,48,64,128", "--n", "2048,2048,1024,512,512,256", "--batch", "128,64,32,16,16,8"]
env = dict(os.environ, XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
free = list(CORES)


def run(job):
    variant, seed = job
    core = free.pop(0)
    try:
        out = f"runs/{variant}_s{seed}"
        os.makedirs(out, exist_ok=True)
        with open(f"{out}/log.txt", "a") as log:
            if not os.path.exists(f"{out}/model.pkl"):
                subprocess.run(["taskset", "-c", str(core), sys.executable, "-m", "scaleflow.train",
                                f"configs/{variant}.json", "--out", out, "--seed", str(seed)],
                               stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            if not os.path.exists(f"{out}/eval.json") or "--force-eval" in sys.argv:
                subprocess.run(["taskset", "-c", str(core), sys.executable, "-m", "scaleflow.evaluate", out, *EVAL],
                               stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        print(time.strftime("%H:%M:%S"), "done", out, flush=True)
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "FAILED", job, e, flush=True)
    finally:
        free.append(core)


with ThreadPoolExecutor(len(CORES)) as ex:
    list(ex.map(run, JOBS))
print("ALL DONE", flush=True)
