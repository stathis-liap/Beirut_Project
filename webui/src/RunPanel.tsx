import { useEffect, useRef, useState } from 'react'
import { useStore } from './store'
import { btn, h3, inputStyle, sliderLabel } from './panelStyles'

interface StormSummary {
  name: string
  total_mm: number
  duration: number
  return_period_yr?: number
  steps: number[][]
}

interface RunMeta {
  closure_rel: number
  vol_rain_m3: number
  vol_infiltrated_m3: number
  vol_outflow_m3: number
  vol_drained_m3: number
  vol_stored_end_m3: number
}

interface ProgressMsg {
  state: 'queued' | 'running' | 'done' | 'error'
  t?: number
  duration?: number
  pct?: number
  eta_s?: number | null
  storage_m3?: number
  outflow_m3?: number
  max_h?: number
  meta?: RunMeta
  error?: string
}

const BIN_S = 300 // 5-minute bins

function presetBins(n: number, totalMm: number, shape: 'uniform' | 'front' | 'peak'): number[] {
  const hoursPerBin = BIN_S / 3600
  let weights: number[]
  if (shape === 'uniform') {
    weights = Array(n).fill(1)
  } else if (shape === 'front') {
    weights = Array.from({ length: n }, (_, i) => n - i)
  } else {
    const center = n * 0.55
    weights = Array.from({ length: n }, (_, i) => Math.max(0.3, 3 - Math.abs(i - center) * (3 / Math.max(n / 2, 1))))
  }
  const sumW = weights.reduce((a, b) => a + b, 0) || 1
  return weights.map((w) => ((w / sumW) * totalMm) / hoursPerBin)
}

export default function RunPanel() {
  const designName = useStore((s) => s.designName)
  const setSharedRunProgress = useStore((s) => s.setRunProgress)

  const [storms, setStorms] = useState<StormSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [customMode, setCustomMode] = useState(false)
  const [durationMin, setDurationMin] = useState(60)
  const [totalMm, setTotalMm] = useState(25)
  const [bins, setBins] = useState<number[]>(() => presetBins(12, 25, 'peak'))
  const [customName, setCustomName] = useState('')

  const [progress, setProgress] = useState<{ runId: string; msg: ProgressMsg } | null>(null)
  const [runError, setRunError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    ;(async () => {
      const names: string[] = await (await fetch('/api/storms')).json()
      const details = await Promise.all(names.map((n) => fetch(`/api/storms/${n}`).then((r) => r.json())))
      setStorms(details)
      // default to the frequent (t2) storm if present - the most typical
      // first thing a planner would want to try - else the first available.
      if (details.length) {
        const preferred = details.find((s) => s.name === 't2') ?? details[0]
        setSelected((prev) => prev ?? preferred.name)
      }
    })()
    return () => {
      wsRef.current?.close()
    }
  }, [])

  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, c.width, c.height)
    const maxV = Math.max(10, ...bins)
    const bw = c.width / bins.length
    bins.forEach((v, i) => {
      const h = (v / maxV) * (c.height - 10)
      ctx.fillStyle = '#3b82f6'
      ctx.fillRect(i * bw + 1, c.height - h, bw - 2, h)
    })
  }, [bins])

  function onBarPointer(e: React.PointerEvent<HTMLCanvasElement>) {
    if (e.buttons !== 1) return
    const rect = canvasRef.current!.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const bw = rect.width / bins.length
    const i = Math.min(bins.length - 1, Math.max(0, Math.floor(x / bw)))
    const maxV = Math.max(10, ...bins) * 1.2
    const v = Math.max(0, Math.min(maxV, ((rect.height - y) / rect.height) * maxV))
    const next = bins.slice()
    next[i] = Math.round(v * 10) / 10
    setBins(next)
  }

  function applyPreset(shape: 'uniform' | 'front' | 'peak') {
    setBins(presetBins(Math.max(1, Math.round((durationMin * 60) / BIN_S)), totalMm, shape))
  }

  function rescaleDuration(newDurationMin: number) {
    setDurationMin(newDurationMin)
    const n = Math.max(1, Math.round((newDurationMin * 60) / BIN_S))
    setBins((prev) => Array.from({ length: n }, (_, i) => prev[Math.min(i, prev.length - 1)] ?? 0))
  }

  function rescaleTotal(newTotalMm: number) {
    const curTotalMm = bins.reduce((s, v) => s + v, 0) * (BIN_S / 3600)
    const factor = curTotalMm > 0 ? newTotalMm / curTotalMm : 1
    setTotalMm(newTotalMm)
    setBins(bins.map((v) => v * factor))
  }

  async function saveCustomStorm() {
    const name = customName.trim()
    if (!name) return
    const steps = bins.map((mmh, i) => [i * BIN_S, (i + 1) * BIN_S, Math.round(mmh * 100) / 100])
    const r = await fetch('/api/storms', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, steps }),
    })
    if (!r.ok) {
      alert('failed to save storm: ' + (await r.text()))
      return
    }
    const saved: StormSummary = await r.json()
    setStorms((prev) => [...prev.filter((s) => s.name !== saved.name), saved])
    setSelected(saved.name)
    setCustomMode(false)
  }

  async function runNow() {
    if (!designName || !selected) return
    setRunError(null)
    const r = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ design: designName, storm: selected }),
    })
    if (!r.ok) {
      const body = await r.json().catch(() => ({ detail: r.statusText }))
      setRunError(body.detail)
      return
    }
    const { run_id } = await r.json()
    setProgress({ runId: run_id, msg: { state: 'queued' } })
    setSharedRunProgress({ runId: run_id, storm: selected, state: 'queued' })
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${proto}//${location.host}/api/run/${run_id}/progress`)
    wsRef.current = ws
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as ProgressMsg
      setProgress({ runId: run_id, msg })
      setSharedRunProgress({
        runId: run_id,
        storm: selected,
        state: msg.state,
        pct: msg.pct,
        t: msg.t,
        duration: msg.duration,
        eta_s: msg.eta_s,
        error: msg.error,
      })
      if (msg.state === 'done' || msg.state === 'error') {
        setTimeout(() => setSharedRunProgress(null), 5000)
      }
    }
    ws.onerror = () => {
      setRunError('progress connection lost')
      setSharedRunProgress(null)
    }
  }

  const isRunning = progress?.msg.state === 'queued' || progress?.msg.state === 'running'
  const selectedStorm = storms.find((s) => s.name === selected)

  return (
    <>
      <section>
        <h3 style={h3}>Rain</h3>
        {!customMode ? (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {storms.map((s) => (
                <button
                  key={s.name}
                  style={{ ...btn, background: selected === s.name ? '#3b82f6' : btn.background, textAlign: 'left' }}
                  onClick={() => setSelected(s.name)}
                >
                  {s.return_period_yr ? `${s.return_period_yr}-year storm` : s.name} — {s.total_mm.toFixed(1)} mm over{' '}
                  {(s.duration / 60).toFixed(0)} min
                </button>
              ))}
            </div>
            <button style={{ ...btn, width: '100%', marginTop: 8 }} onClick={() => setCustomMode(true)}>
              + Custom storm
            </button>
          </>
        ) : (
          <div style={{ border: '1px solid #333', borderRadius: 4, padding: 8 }}>
            <input
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              placeholder="storm name"
              style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' }}
            />
            <label style={sliderLabel}>
              Rain duration: {durationMin} min
              <input type="range" min={30} max={180} step={5} value={durationMin} onChange={(e) => rescaleDuration(Number(e.target.value))} />
            </label>
            <label style={sliderLabel}>
              Total rain: {totalMm.toFixed(1)} mm
              <input type="range" min={5} max={150} step={1} value={totalMm} onChange={(e) => rescaleTotal(Number(e.target.value))} />
            </label>
            <div style={{ display: 'flex', gap: 4, marginTop: 6 }}>
              <button style={btn} onClick={() => applyPreset('uniform')}>
                Uniform
              </button>
              <button style={btn} onClick={() => applyPreset('front')}>
                Front-loaded
              </button>
              <button style={btn} onClick={() => applyPreset('peak')}>
                Peak
              </button>
            </div>
            <canvas
              ref={canvasRef}
              width={260}
              height={100}
              onPointerDown={onBarPointer}
              onPointerMove={onBarPointer}
              style={{ background: '#111', cursor: 'crosshair', marginTop: 8, width: '100%' }}
            />
            <div style={{ opacity: 0.7, marginTop: 4 }}>drag on the bars to set intensity per 5-min step (mm/h)</div>
            <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
              <button style={btn} onClick={saveCustomStorm}>
                Save storm
              </button>
              <button style={btn} onClick={() => setCustomMode(false)}>
                Cancel
              </button>
            </div>
          </div>
        )}
      </section>

      <section>
        <h3 style={h3}>Run</h3>
        <button
          style={{ ...btn, width: '100%' }}
          onClick={runNow}
          disabled={!selected || isRunning}
          title={!selected ? 'pick a storm above first' : isRunning ? 'a simulation is already in progress' : undefined}
        >
          {isRunning ? 'Running…' : `Run ${selectedStorm?.name ?? ''}`}
        </button>
        {runError && <div style={{ color: '#f87171', marginTop: 6 }}>{runError}</div>}
        {progress && (
          <div style={{ marginTop: 8 }}>
            {progress.msg.state === 'queued' && <div>Queued…</div>}
            {progress.msg.state === 'running' && (
              <>
                <div>
                  Simulating: {progress.msg.pct?.toFixed(0)}% (t={progress.msg.t?.toFixed(0)}s / {progress.msg.duration?.toFixed(0)}s)
                </div>
                <progress value={progress.msg.pct} max={100} style={{ width: '100%' }} />
                <div style={{ opacity: 0.8 }}>
                  storage {progress.msg.storage_m3?.toFixed(0)} m³ · outflow {progress.msg.outflow_m3?.toFixed(0)} m³ · max depth{' '}
                  {progress.msg.max_h?.toFixed(2)} m
                </div>
                {progress.msg.eta_s != null && <div style={{ opacity: 0.8 }}>ETA {progress.msg.eta_s.toFixed(0)}s</div>}
              </>
            )}
            {progress.msg.state === 'done' && progress.msg.meta && (
              <div>
                <div style={{ color: '#4ade80' }}>Done.</div>
                <div>water accounting error: {(progress.msg.meta.closure_rel * 100).toFixed(4)}%</div>
                <div style={{ opacity: 0.8, marginTop: 4 }}>
                  rain {progress.msg.meta.vol_rain_m3.toFixed(0)} m³ = infiltrated {progress.msg.meta.vol_infiltrated_m3.toFixed(0)} + outflow{' '}
                  {progress.msg.meta.vol_outflow_m3.toFixed(0)} + stored {progress.msg.meta.vol_stored_end_m3.toFixed(0)}
                </div>
              </div>
            )}
            {progress.msg.state === 'error' && <div style={{ color: '#f87171' }}>Run failed: {progress.msg.error}</div>}
          </div>
        )}
      </section>
    </>
  )
}
