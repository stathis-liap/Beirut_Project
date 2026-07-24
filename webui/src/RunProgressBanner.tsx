import { useStore } from './store'

/** Always-visible progress banner for the currently running simulation,
 * regardless of which tab (2D/3D/Results) is active - the run panel itself
 * lives at the bottom of a scrollable sidebar, easy to miss. */
export default function RunProgressBanner() {
  const p = useStore((s) => s.runProgress)
  if (!p) return null

  const pct = p.pct ?? 0

  return (
    <div
      style={{
        position: 'fixed',
        top: 180, // clears the 3D view's vertical-exaggeration panel in the top-right
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 50,
        background: 'rgba(20,20,20,0.95)',
        color: '#fff',
        font: '13px sans-serif',
        padding: '10px 18px',
        borderRadius: 6,
        border: '1px solid #3b82f6',
        boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
        minWidth: 320,
      }}
    >
      {p.state === 'queued' && <div>Simulating {p.storm}: queued…</div>}
      {p.state === 'running' && (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <span>
              Simulating {p.storm}: {pct.toFixed(0)}%
            </span>
            <span style={{ opacity: 0.8 }}>{p.eta_s != null ? `ETA ${formatEta(p.eta_s)}` : ''}</span>
          </div>
          <div style={{ height: 6, background: '#333', borderRadius: 3, marginTop: 6, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${pct}%`, background: '#3b82f6', transition: 'width 0.3s' }} />
          </div>
          {p.t != null && p.duration != null && (
            <div style={{ opacity: 0.7, marginTop: 4 }}>
              t = {p.t.toFixed(0)}s / {p.duration.toFixed(0)}s
            </div>
          )}
        </>
      )}
      {p.state === 'done' && <div style={{ color: '#4ade80' }}>✓ {p.storm} finished — see the Results tab.</div>}
      {p.state === 'error' && <div style={{ color: '#f87171' }}>✗ {p.storm} failed: {p.error}</div>}
    </div>
  )
}

function formatEta(s: number): string {
  if (s < 60) return `${Math.round(s)}s`
  const m = Math.floor(s / 60)
  const rem = Math.round(s % 60)
  return `${m}m ${rem}s`
}
