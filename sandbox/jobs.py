"""Single-worker GPU job queue for sandbox simulation runs.

Only one simulation may run at a time (one GPU). The worker thread bakes the
design, calls flood_gpu.simulate with a progress callback, and writes the
usual flood_gpu output files plus a run.json that ties the result back to
the exact design (by content hash) and storm that produced it.
"""
import datetime
import json
import os
import queue
import threading

import numpy as np

from flood_gpu import simulate

from sandbox.baking import bake
from sandbox.state import SANDBOX_DIR, design_sha256

RUNS_DIR = os.path.join(SANDBOX_DIR, "runs")


def _run_id(design_name, storm_name):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{design_name}__{storm_name}__{stamp}"


class Job:
    def __init__(self, run_id, design_name, storm_name, storm_json, duration, save_every):
        self.run_id = run_id
        self.design_name = design_name
        self.storm_name = storm_name
        self.storm_json = storm_json
        self.duration = duration
        self.save_every = save_every
        self.run_dir = os.path.join(RUNS_DIR, run_id)
        self.state = "queued"  # queued | running | done | error
        self.latest = {"state": "queued"}
        self.error = None


class JobQueue:
    def __init__(self, base, design_store):
        self.base = base
        self.design_store = design_store
        self.jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._lock = threading.Lock()
        os.makedirs(RUNS_DIR, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()

    def enqueue(self, design_name, storm_name, storm_json, duration, save_every=60.0):
        if design_name not in self.design_store.designs:
            raise KeyError(f"unknown design '{design_name}'")
        with self._lock:
            for j in self.jobs.values():
                if j.design_name == design_name and j.state in ("queued", "running"):
                    raise RuntimeError(f"design '{design_name}' already has a run queued or in progress")
            queued_behind = sum(1 for j in self.jobs.values() if j.state == "queued")
        run_id = _run_id(design_name, storm_name)
        job = Job(run_id, design_name, storm_name, storm_json, duration, save_every)
        with self._lock:
            self.jobs[run_id] = job
        self._queue.put(job)
        return job, queued_behind

    def get(self, run_id) -> Job:
        if run_id not in self.jobs:
            raise KeyError(run_id)
        return self.jobs[run_id]

    def _worker(self):
        while True:
            job = self._queue.get()
            try:
                job.state = "running"
                job.latest = {"state": "running"}
                self._run(job)
                job.state = "done"
            except Exception as e:
                job.state = "error"
                job.error = str(e)
                job.latest = {**job.latest, "state": "error", "error": str(e)}
            self._queue.task_done()

    def _run(self, job):
        design = self.design_store.get(job.design_name)
        dem, man, infil = bake(self.base, design)
        os.makedirs(job.run_dir, exist_ok=True)

        run_meta = {
            "run_id": job.run_id, "design": job.design_name, "storm": job.storm_name,
            "design_sha256": design_sha256(design), "storm_json": job.storm_json,
            "label": "", "notes": "", "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        with open(os.path.join(job.run_dir, "run.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
        # keep the exact material footprint used, so metrics/comparisons stay
        # valid even if the source design is later edited or deleted.
        np.save(os.path.join(job.run_dir, "material.npy"), design.material)

        def progress_cb(t, duration, stats):
            eta_s = stats["wall_s"] * (duration / t - 1) if t > 0 else None
            job.latest = {
                "state": "running", "t": t, "duration": duration,
                "pct": round(100.0 * t / max(duration, 1e-9), 1),
                "eta_s": round(eta_s, 1) if eta_s is not None else None,
                **stats,
            }

        simulate(
            dem, self.base.t["res"], job.storm_json["steps"], job.duration, job.run_dir,
            manning=man, infil_mmh=infil, valid=self.base.masks["valid"],
            water=self.base.masks["water"], rain_weight=self.base.rain_w,
            gauges=self.base.gauges, save_every=job.save_every, device="auto",
            save_frames=True, progress=False, progress_cb=progress_cb,
        )
        with open(os.path.join(job.run_dir, "run_meta.json")) as f:
            job.latest = {"state": "done", "meta": json.load(f)}
