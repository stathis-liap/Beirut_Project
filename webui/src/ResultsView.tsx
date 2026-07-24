import { useEffect, useState } from 'react'
import {
  deleteRun,
  getCompare,
  getRun,
  getRunFrames,
  listRuns,
  patchRun,
  type CompareResult,
  type RunDetail,
  type RunSummary,
} from './api'
import ResultMap from './ResultMap'
import { btn, h3, inputStyle } from './panelStyles'

type Overlay = 'animation' | 'maxdepth' | 'hazard'
type Mode = 'library' | 'view' | 'compare'

export default function ResultsView() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [mode, setMode] = useState<Mode>('library')
  const [activeRun, setActiveRun] = useState<string | null>(null)
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [overlay, setOverlay] = useState<Overlay>('maxdepth')
  const [frames, setFrames] = useState<number[]>([])
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const [compareA, setCompareA] = useState<string>('')
  const [compareB, setCompareB] = useState<string>('')
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null)

  async function refresh() {
    setRuns(await listRuns())
  }
  useEffect(() => {
    refresh()
  }, [])

  async function openRun(runId: string) {
    setActiveRun(runId)
    setMode('view')
    setOverlay('maxdepth')
    setFrameIdx(0)
    setPlaying(false)
    setDetail(null)
    const [d, fr] = await Promise.all([getRun(runId), getRunFrames(runId)])
    setDetail(d)
    setFrames(fr)
  }

  useEffect(() => {
    if (!playing || frames.length === 0) return
    const id = setInterval(() => setFrameIdx((i) => (i + 1) % frames.length), 100)
    return () => clearInterval(id)
  }, [playing, frames])

  async function doCompare() {
    if (!compareA || !compareB) return
    setCompareResult(await getCompare(compareA, compareB))
  }

  async function rename(runId: string) {
    const label = prompt('Label for this run:')
    if (label === null) return
    await patchRun(runId, { label })
    refresh()
  }

  async function remove(runId: string) {
    if (!confirm(`Delete run "${runId}"? This cannot be undone.`)) return
    await deleteRun(runId)
    if (activeRun === runId) {
      setActiveRun(null)
      setMode('library')
    }
    refresh()
  }

  const overlaySrc = !activeRun
    ? null
    : overlay === 'animation' && frames.length > 0
      ? `/api/runs/${activeRun}/frame/${frames[frameIdx]}.png`
      : overlay === 'maxdepth'
        ? `/api/runs/${activeRun}/max_depth.png`
        : overlay === 'hazard'
          ? `/api/runs/${activeRun}/hazard.png`
          : null

  const compareSrc = compareResult && compareA && compareB ? `/api/compare/diff.png?a=${compareA}&b=${compareB}` : null

  return (
    <div style={{ display: 'flex', height: '100%' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        {mode === 'view' && activeRun ? (
          <ResultMap overlaySrc={overlaySrc} />
        ) : mode === 'compare' && compareSrc ? (
          <ResultMap overlaySrc={compareSrc} />
        ) : (
          <div style={{ padding: 20, color: '#ddd' }}>Select a run from the panel on the right.</div>
        )}
      </div>
      <div
        style={{
          width: 320,
          flexShrink: 0,
          background: '#1a1a1a',
          color: '#ddd',
          borderLeft: '1px solid #333',
          overflowY: 'auto',
          padding: 12,
          font: '13px sans-serif',
        }}
      >
        <div style={{ display: 'flex', gap: 4, marginBottom: 12 }}>
          <button style={{ ...btn, background: mode === 'library' ? '#3b82f6' : btn.background }} onClick={() => setMode('library')}>
            Library
          </button>
          <button style={{ ...btn, background: mode === 'compare' ? '#3b82f6' : btn.background }} onClick={() => setMode('compare')}>
            Compare
          </button>
        </div>

        {mode === 'library' && (
          <div>
            <h3 style={h3}>Runs</h3>
            {runs.length === 0 && <div style={{ opacity: 0.6 }}>no runs yet — go run a simulation from the 2D tab</div>}
            {runs.map((r) => (
              <div key={r.run_id} style={{ border: '1px solid #333', borderRadius: 4, padding: 8, marginBottom: 8 }}>
                <div style={{ fontWeight: 'bold' }}>{r.label || r.run_id}</div>
                <div style={{ opacity: 0.8 }}>
                  {r.design} · {r.storm} · {new Date(r.created).toLocaleString()}
                </div>
                {r.finished ? (
                  <div style={{ opacity: 0.8 }}>
                    rain {r.vol_rain_m3?.toFixed(0)} m³ · water accounting error {((r.closure_rel ?? 0) * 100).toFixed(3)}%
                  </div>
                ) : (
                  <div style={{ color: '#facc15' }}>not finished yet</div>
                )}
                {r.notes && <div style={{ opacity: 0.7, fontStyle: 'italic', marginTop: 2 }}>{r.notes}</div>}
                <div style={{ display: 'flex', gap: 4, marginTop: 6 }}>
                  <button style={btn} onClick={() => openRun(r.run_id)}>
                    View
                  </button>
                  <button style={btn} onClick={() => rename(r.run_id)}>
                    Rename
                  </button>
                  <button style={{ ...btn, color: '#f87171' }} onClick={() => remove(r.run_id)}>
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {mode === 'view' && detail && (
          <div>
            <h3 style={h3}>{detail.run.label || detail.run.run_id}</h3>
            <div style={{ display: 'flex', gap: 4, marginBottom: 8, flexWrap: 'wrap' }}>
              <button style={{ ...btn, background: overlay === 'animation' ? '#3b82f6' : btn.background }} onClick={() => setOverlay('animation')}>
                Animation
              </button>
              <button style={{ ...btn, background: overlay === 'maxdepth' ? '#3b82f6' : btn.background }} onClick={() => setOverlay('maxdepth')}>
                Max depth
              </button>
              <button style={{ ...btn, background: overlay === 'hazard' ? '#3b82f6' : btn.background }} onClick={() => setOverlay('hazard')}>
                Hazard
              </button>
            </div>
            {overlay === 'animation' && frames.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <input
                  type="range"
                  min={0}
                  max={frames.length - 1}
                  value={frameIdx}
                  onChange={(e) => setFrameIdx(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>t = {frames[frameIdx]}s</span>
                  <button style={btn} onClick={() => setPlaying((p) => !p)}>
                    {playing ? 'Pause' : 'Play'}
                  </button>
                </div>
              </div>
            )}
            <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div>flooded street area: {detail.metrics.flooded_streets_m2} m²</div>
              <div>flooded on corridor: {detail.metrics.flooded_on_ribbon_m2} m²</div>
              <div>p99 depth on corridor: {detail.metrics.p99_depth_ribbon_cm} cm</div>
              <div>infiltrated: {detail.metrics.infil_pct}% of rain</div>
              <div>outflow: {detail.metrics.vol_outflow_m3.toFixed(0)} m³</div>
              <div>water accounting error: {(detail.metrics.closure_rel * 100).toFixed(4)}%</div>
            </div>
          </div>
        )}

        {mode === 'compare' && (
          <div>
            <h3 style={h3}>Compare two runs</h3>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 2, marginBottom: 8 }}>
              Before (A)
              <select style={{ ...inputStyle, width: '100%' }} value={compareA} onChange={(e) => setCompareA(e.target.value)}>
                <option value="">select…</option>
                {runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.label || r.run_id}
                  </option>
                ))}
              </select>
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 2, marginBottom: 8 }}>
              After (B)
              <select style={{ ...inputStyle, width: '100%' }} value={compareB} onChange={(e) => setCompareB(e.target.value)}>
                <option value="">select…</option>
                {runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.label || r.run_id}
                  </option>
                ))}
              </select>
            </label>
            <button style={{ ...btn, width: '100%' }} onClick={doCompare} disabled={!compareA || !compareB}>
              Compare
            </button>
            {compareResult && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontWeight: 'bold', marginBottom: 8 }}>{headline(compareResult)}</div>
                <div style={{ opacity: 0.7, marginBottom: 8 }}>diff map: blue = shallower in B, red = deeper in B</div>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left' }}>metric</th>
                      <th style={{ textAlign: 'right' }}>A</th>
                      <th style={{ textAlign: 'right' }}>B</th>
                      <th style={{ textAlign: 'right' }}>Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(compareResult.deltas).map(([k, v]) => (
                      <tr key={k} style={{ borderTop: '1px solid #333' }}>
                        <td>{k}</td>
                        <td style={{ textAlign: 'right' }}>{v.a}</td>
                        <td style={{ textAlign: 'right' }}>{v.b}</td>
                        <td style={{ textAlign: 'right' }}>{v.delta}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function headline(c: CompareResult): string {
  const before = c.deltas.flooded_streets_m2.a
  const after = c.deltas.flooded_streets_m2.b
  if (before <= 0) return 'No flooding in run A to compare against.'
  const pct = Math.round(100 * (1 - after / before))
  if (pct > 0) return `Run B reduces flooded street area by ${pct}% versus run A.`
  if (pct < 0) return `Run B increases flooded street area by ${-pct}% versus run A.`
  return 'No meaningful change in flooded street area between the two runs.'
}
