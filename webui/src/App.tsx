import { useState } from 'react'
import MapView from './MapView'
import Scene3D from './Scene3D'
import DesignPanel from './DesignPanel'
import ResultsView from './ResultsView'
import RunProgressBanner from './RunProgressBanner'
import { useStore } from './store'

type View = '2d' | '3d' | 'results'

const STEPS = [
  { n: 1, label: 'Design' },
  { n: 2, label: 'Rain' },
  { n: 3, label: 'Run & results' },
]

function App() {
  const [view, setView] = useState<View>('2d')
  const designName = useStore((s) => s.designName)

  const currentStep = !designName ? 1 : view === 'results' ? 3 : 2

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header
        style={{
          padding: '8px 16px',
          borderBottom: '1px solid #333',
          background: '#1a1a1a',
          color: '#fff',
          display: 'flex',
          alignItems: 'center',
          gap: 24,
        }}
      >
        <strong>Al-Masar Sandbox</strong>
        <div style={{ display: 'flex', gap: 4 }}>
          {(['2d', '3d', 'results'] as View[]).map((v) => (
            <button key={v} onClick={() => setView(v)} style={{ ...tabBtn, background: view === v ? '#3b82f6' : tabBtn.background }}>
              {v === '2d' ? '2D' : v === '3d' ? '3D' : 'Results'}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8, font: '12px sans-serif', opacity: 0.9 }}>
          {STEPS.map((s, i) => (
            <span key={s.n} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span
                style={{
                  color: currentStep === s.n ? '#fff' : currentStep > s.n ? '#4ade80' : '#888',
                  fontWeight: currentStep === s.n ? 'bold' : 'normal',
                }}
              >
                {currentStep > s.n ? '✓' : `${s.n}.`} {s.label}
              </span>
              {i < STEPS.length - 1 && <span style={{ color: '#555' }}>→</span>}
            </span>
          ))}
        </div>
      </header>
      <RunProgressBanner />
      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <div style={{ flex: 1, minWidth: 0, display: view === '2d' ? 'block' : 'none' }}>
          <MapView />
        </div>
        {view === '3d' && (
          <div style={{ flex: 1, minWidth: 0 }}>
            <Scene3D />
          </div>
        )}
        {view === 'results' && (
          <div style={{ flex: 1, minWidth: 0 }}>
            <ResultsView />
          </div>
        )}
        {view !== 'results' && <DesignPanel />}
      </div>
    </div>
  )
}

const tabBtn: React.CSSProperties = {
  background: '#2a2a2a',
  color: '#ddd',
  border: '1px solid #444',
  borderRadius: 4,
  padding: '4px 10px',
  cursor: 'pointer',
  font: '12px sans-serif',
}

export default App
