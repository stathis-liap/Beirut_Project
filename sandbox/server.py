#!/usr/bin/env python3
"""FastAPI backend for the interactive green-corridor sandbox.

Run (dev): /home/stathisliap/Work/.venv/bin/uvicorn sandbox.server:app --reload --port 8008
Run (prod, after `npm run build` in webui/):
    /home/stathisliap/Work/.venv/bin/uvicorn sandbox.server:app --port 8008
"""
import asyncio
import base64
import contextlib
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sandbox.encode import dem_bin, depth_png, diff_png, hazard_png, masks_png
from sandbox.jobs import JobQueue, RUNS_DIR
from sandbox.metrics import get_or_compute_metrics
from sandbox.state import (base, clamp_materials, design_store, get_storm,
                            list_storms, save_storm, MAX_PATCH_CELLS)

job_queue = JobQueue(base, design_store)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    design_store.flush_dirty()


app = FastAPI(title="Al-Masar Sandbox", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DAY = "public, max-age=86400"


class CreateDesignBody(BaseModel):
    name: str
    template: str = "blank"


class PatchBody(BaseModel):
    x: int
    y: int
    w: int
    h: int
    material_b64: str | None = None
    dem_delta_b64: str | None = None


class LockBody(BaseModel):
    unlocked: bool


class MaterialItem(BaseModel):
    id: int
    label: str
    color: str
    infil_mmh: float
    manning_n: float
    depression_m: float
    builtin: bool = False


class MaterialsBody(BaseModel):
    materials: list[MaterialItem]


class StormBody(BaseModel):
    name: str
    steps: list[list[float]]  # [[t0_s, t1_s, mm_h], ...]


class RunBody(BaseModel):
    design: str
    storm: str
    duration: float | None = None
    save_every: float = 60.0


class RunPatchBody(BaseModel):
    label: str | None = None
    notes: str | None = None


def _require_design(name: str):
    if name not in design_store.designs:
        raise HTTPException(404, f"unknown design '{name}'")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    return {
        "transform": base.t,
        "width": base.w,
        "height": base.h,
        "res": base.t["res"],
        "gauges": base.gauges,
        "storms": list_storms(),
        "dem_min": base.dem_min,
        "dem_scale": base.dem_scale,
        "hazard_bounds": [0.75, 1.25, 2.0],
    }


@app.get("/api/terrain/ortho.png")
def terrain_ortho():
    return FileResponse(os.path.join(base.terrain, "ortho.png"), headers={"Cache-Control": DAY})


@app.get("/api/terrain/hillshade.png")
def terrain_hillshade():
    return FileResponse(os.path.join(base.terrain, "hillshade.png"), headers={"Cache-Control": DAY})


@app.get("/api/terrain/dem.bin")
def terrain_dem_bin():
    body = dem_bin(base.dem, base.masks["valid"], base.dem_min, base.dem_scale)
    return Response(body, media_type="application/octet-stream", headers={"Cache-Control": DAY})


@app.get("/api/terrain/masks.png")
def terrain_masks_png():
    body = masks_png(base.masks, base.zone)
    return Response(body, media_type="image/png", headers={"Cache-Control": DAY})


@app.get("/api/designs")
def list_designs():
    return design_store.list()


@app.post("/api/designs")
def create_design(body: CreateDesignBody):
    if body.name in design_store.designs:
        raise HTTPException(409, f"design '{body.name}' already exists")
    try:
        design = design_store.create(body.name, body.template)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return design.to_json()


@app.get("/api/designs/{name}")
def get_design(name: str):
    _require_design(name)
    return design_store.get(name).to_json()


@app.get("/api/designs/{name}/material.bin")
def get_design_material(name: str):
    _require_design(name)
    body = design_store.get(name).material.tobytes()
    return Response(body, media_type="application/octet-stream")


@app.get("/api/designs/{name}/dem_delta.bin")
def get_design_dem_delta(name: str):
    _require_design(name)
    body = design_store.get(name).dem_delta.tobytes()
    return Response(body, media_type="application/octet-stream")


@app.post("/api/designs/{name}/patch")
def patch_design(name: str, body: PatchBody):
    _require_design(name)
    if body.w <= 0 or body.h <= 0:
        raise HTTPException(422, "w and h must be positive")
    if body.w * body.h > MAX_PATCH_CELLS:
        raise HTTPException(413, f"patch of {body.w * body.h} cells exceeds the {MAX_PATCH_CELLS} limit")

    material = None
    if body.material_b64:
        raw = base64.b64decode(body.material_b64)
        if len(raw) != body.w * body.h * 2:
            raise HTTPException(422, "material_b64 size does not match w*h uint16")
        material = np.frombuffer(raw, dtype=np.uint16).reshape(body.h, body.w)

    dem_delta = None
    if body.dem_delta_b64:
        raw = base64.b64decode(body.dem_delta_b64)
        if len(raw) != body.w * body.h * 4:
            raise HTTPException(422, "dem_delta_b64 size does not match w*h float32")
        dem_delta = np.frombuffer(raw, dtype=np.float32).reshape(body.h, body.w)

    applied = design_store.patch(name, body.x, body.y, body.w, body.h, material, dem_delta)
    return {"applied_cells": applied}


@app.put("/api/designs/{name}/materials")
def put_materials(name: str, body: MaterialsBody):
    _require_design(name)
    clamped = clamp_materials([m.model_dump() for m in body.materials])
    design_store.set_materials(name, clamped)
    return clamped


@app.put("/api/designs/{name}/lock")
def put_lock(name: str, body: LockBody):
    _require_design(name)
    design_store.set_lock(name, body.unlocked)
    return design_store.get(name).to_json()["design"]


@app.post("/api/designs/{name}/save")
def save_design(name: str):
    _require_design(name)
    design_store.save(name)
    return {"saved": True}


@app.delete("/api/designs/{name}", status_code=204)
def delete_design(name: str):
    _require_design(name)
    design_store.delete(name)


@app.get("/api/storms")
def get_storms():
    return list_storms()


@app.get("/api/storms/{name}")
def get_storm_endpoint(name: str):
    try:
        return get_storm(name)
    except KeyError:
        raise HTTPException(404, f"unknown storm '{name}'")


@app.post("/api/storms")
def post_storm(body: StormBody):
    if not body.steps:
        raise HTTPException(422, "steps must be non-empty")
    for step in body.steps:
        if len(step) != 3:
            raise HTTPException(422, "each step must be [t0_s, t1_s, mm_h]")
    return save_storm({"name": body.name, "steps": body.steps})


@app.post("/api/run")
def post_run(body: RunBody):
    _require_design(body.design)
    try:
        storm_json = get_storm(body.storm)
    except KeyError:
        raise HTTPException(404, f"unknown storm '{body.storm}'")
    duration = body.duration or storm_json["duration"]
    try:
        job, queued_behind = job_queue.enqueue(body.design, body.storm, storm_json, duration, body.save_every)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"run_id": job.run_id, "queued_behind": queued_behind}


@app.websocket("/api/run/{run_id}/progress")
async def run_progress(ws: WebSocket, run_id: str):
    await ws.accept()
    try:
        job = job_queue.get(run_id)
    except KeyError:
        await ws.close(code=4404)
        return
    last_sent = None
    try:
        while True:
            if job.latest != last_sent:
                await ws.send_json(job.latest)
                last_sent = job.latest
            if job.state in ("done", "error"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    await ws.close()


def _run_dir(run_id: str) -> str:
    d = os.path.join(RUNS_DIR, run_id)
    if not os.path.isdir(d):
        raise HTTPException(404, f"unknown run '{run_id}'")
    return d


def _load_run_json(run_id: str) -> dict:
    d = _run_dir(run_id)
    with open(os.path.join(d, "run.json")) as f:
        return json.load(f)


def _load_run_meta(run_id: str) -> dict:
    d = _run_dir(run_id)
    path = os.path.join(d, "run_meta.json")
    if not os.path.exists(path):
        raise HTTPException(409, f"run '{run_id}' has not finished yet")
    with open(path) as f:
        return json.load(f)


def _run_material(run_id: str) -> np.ndarray:
    path = os.path.join(_run_dir(run_id), "material.npy")
    if os.path.exists(path):
        return np.load(path)
    return np.zeros((base.h, base.w), np.uint16)


@app.get("/api/runs")
def list_runs():
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        d = os.path.join(RUNS_DIR, name)
        run_path = os.path.join(d, "run.json")
        if not os.path.isdir(d) or not os.path.exists(run_path):
            continue
        with open(run_path) as f:
            entry = json.load(f)
        meta_path = os.path.join(d, "run_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            entry.update({
                "finished": True,
                "closure_rel": meta.get("closure_rel"),
                "vol_rain_m3": meta.get("vol_rain_m3"),
                "vol_infiltrated_m3": meta.get("vol_infiltrated_m3"),
                "vol_outflow_m3": meta.get("vol_outflow_m3"),
                "vol_stored_end_m3": meta.get("vol_stored_end_m3"),
            })
        else:
            entry["finished"] = False
        out.append(entry)
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = _load_run_json(run_id)
    meta = _load_run_meta(run_id)
    metrics = get_or_compute_metrics(base, _run_material(run_id), _run_dir(run_id))
    return {"run": run, "meta": meta, "metrics": metrics}


@app.get("/api/runs/{run_id}/frames")
def get_run_frames(run_id: str):
    d = _run_dir(run_id)
    times = []
    for fp in sorted(glob.glob(os.path.join(d, "depth_*.npy"))):
        stem = os.path.splitext(os.path.basename(fp))[0]
        times.append(int(stem.split("_")[1]))
    return times


@app.get("/api/runs/{run_id}/frame/{t}.png")
def get_run_frame_png(run_id: str, t: int, vmax: float | None = None):
    d = _run_dir(run_id)
    fp = os.path.join(d, f"depth_{t:06d}.npy")
    if not os.path.exists(fp):
        raise HTTPException(404, f"no frame at t={t}")
    depth = np.load(fp).astype(np.float32)
    if vmax is None:
        vmax = get_or_compute_metrics(base, _run_material(run_id), d)["default_vmax_m"]
    return Response(depth_png(depth, vmax), media_type="image/png")


@app.get("/api/runs/{run_id}/max_depth.png")
def get_run_maxdepth_png(run_id: str, vmax: float | None = None):
    d = _run_dir(run_id)
    depth = np.load(os.path.join(d, "max_depth.npy"))
    if vmax is None:
        vmax = get_or_compute_metrics(base, _run_material(run_id), d)["default_vmax_m"]
    return Response(depth_png(depth, vmax), media_type="image/png")


@app.get("/api/runs/{run_id}/hazard.png")
def get_run_hazard_png(run_id: str):
    d = _run_dir(run_id)
    hz = np.load(os.path.join(d, "max_hazard.npy"))
    md = np.load(os.path.join(d, "max_depth.npy"))
    return Response(hazard_png(hz, md), media_type="image/png")


@app.get("/api/runs/{run_id}/gauges.csv")
def get_run_gauges_csv(run_id: str):
    path = os.path.join(_run_dir(run_id), "gauges.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "no gauges recorded for this run")
    return FileResponse(path, media_type="text/csv")


@app.patch("/api/runs/{run_id}")
def patch_run(run_id: str, body: RunPatchBody):
    d = _run_dir(run_id)
    run = _load_run_json(run_id)
    if body.label is not None:
        run["label"] = body.label
    if body.notes is not None:
        run["notes"] = body.notes
    with open(os.path.join(d, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    return run


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str):
    shutil.rmtree(_run_dir(run_id))


@app.get("/api/compare")
def compare_runs(a: str, b: str):
    ma = get_or_compute_metrics(base, _run_material(a), _run_dir(a))
    mb = get_or_compute_metrics(base, _run_material(b), _run_dir(b))
    deltas = {}
    for k in ma:
        va, vb = ma.get(k), mb.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            deltas[k] = {
                "a": va, "b": vb, "delta": round(vb - va, 3),
                "delta_pct": round(100 * (vb - va) / va, 1) if va else None,
            }
    return {"a": ma, "b": mb, "deltas": deltas}


@app.get("/api/compare/diff.png")
def compare_diff_png(a: str, b: str, vmax: float | None = None):
    depth_a = np.load(os.path.join(_run_dir(a), "max_depth.npy"))
    depth_b = np.load(os.path.join(_run_dir(b), "max_depth.npy"))
    if vmax is None:
        d = depth_b - depth_a
        sig = np.abs(d) > 0.02
        vmax = float(np.percentile(np.abs(d[sig]), 99)) if sig.any() else 0.1
    return Response(diff_png(depth_b, depth_a, vmax), media_type="image/png")


dist = os.path.join(os.path.dirname(__file__), "..", "webui", "dist")
if os.path.isdir(dist):
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
