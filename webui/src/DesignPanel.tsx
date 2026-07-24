import { useEffect, useState } from 'react'
import { useStore, type MaterialDef, type Tool } from './store'
import {
  createDesign,
  deleteDesign,
  getDesign,
  getDesignDemDelta,
  getDesignMaterial,
  listDesigns,
  putLock,
  putMaterials,
  saveDesign,
} from './api'
import { deleteMaterialEverywhere } from './editor'
import { btn, h3, h4, inputStyle, sliderLabel } from './panelStyles'
import RunPanel from './RunPanel'

// Plain-language descriptions for the seven built-in materials (ids 1-7,
// from bake_corridor.PROPS). Custom materials (id >= 100) fall back to
// their own label.
const MATERIAL_TOOLTIPS: Record<number, string> = {
  1: 'Vehicular lane: ordinary asphalt road surface — water mostly runs off rather than soaking in.',
  2: 'Porous bikelane: a paving surface that lets rain pass through into the ground below.',
  3: 'Bioswale: a planted channel that soaks up water and slows it down; it sits below the surrounding pavement.',
  4: 'Porous sidewalk: a walking surface that lets rain soak through instead of pooling.',
  5: 'Garden: planted ground with moderate absorption, filling the space around the street.',
  6: 'Bioretention pond / rain garden: a shallow basin at a low point that captures and holds ponding water.',
  7: 'Terrace: a stepped, planted surface used on steeper ground to slow water down.',
}

function materialTooltip(m: MaterialDef): string {
  return MATERIAL_TOOLTIPS[m.id] ?? m.label
}

const TOOLS: { id: Tool; label: string; hint: string }[] = [
  { id: 'paint', label: 'Paint', hint: 'apply the selected material' },
  { id: 'eraser', label: 'Eraser', hint: 'clear material back to the base terrain' },
  { id: 'raise', label: 'Raise', hint: 'build up the ground (add fill)' },
  { id: 'lower', label: 'Lower', hint: 'dig into the ground (remove material)' },
  { id: 'flatten', label: 'Flatten', hint: 'level the area toward its average height' },
  { id: 'smooth', label: 'Smooth', hint: 'soften sharp sculpted edges' },
]

export default function DesignPanel() {
  const meta = useStore((s) => s.meta)
  const designList = useStore((s) => s.designList)
  const setDesignList = useStore((s) => s.setDesignList)
  const designName = useStore((s) => s.designName)
  const openDesign = useStore((s) => s.openDesign)
  const closeDesign = useStore((s) => s.closeDesign)
  const material = useStore((s) => s.material)
  const materials = useStore((s) => s.materials)
  const setMaterialsList = useStore((s) => s.setMaterialsList)
  const unlocked = useStore((s) => s.unlocked)
  const setUnlockedLocal = useStore((s) => s.setUnlockedLocal)
  const dirty = useStore((s) => s.dirty)
  const setDirty = useStore((s) => s.setDirty)
  const bumpLayerVersion = useStore((s) => s.bumpLayerVersion)
  const tool = useStore((s) => s.tool)
  const setTool = useStore((s) => s.setTool)
  const brushRadiusM = useStore((s) => s.brushRadiusM)
  const setBrushRadiusM = useStore((s) => s.setBrushRadiusM)
  const brushStrengthMps = useStore((s) => s.brushStrengthMps)
  const setBrushStrengthMps = useStore((s) => s.setBrushStrengthMps)
  const brushSoftness = useStore((s) => s.brushSoftness)
  const setBrushSoftness = useStore((s) => s.setBrushSoftness)
  const activeMaterialId = useStore((s) => s.activeMaterialId)
  const setActiveMaterialId = useStore((s) => s.setActiveMaterialId)
  const undoCount = useStore((s) => s.undoCount)
  const redoCount = useStore((s) => s.redoCount)
  const undo = useStore((s) => s.undo)
  const redo = useStore((s) => s.redo)
  const overlayMode = useStore((s) => s.overlayMode)
  const setOverlayMode = useStore((s) => s.setOverlayMode)
  const elevationRangeM = useStore((s) => s.elevationRangeM)

  const [busy, setBusy] = useState<string | null>(null)
  const [newName, setNewName] = useState('')

  useEffect(() => {
    ;(async () => {
      const list = await listDesigns()
      setDesignList(list)
      // first-run default: give a non-technical user something to look at
      // immediately instead of an empty picker.
      if (list.length === 0 && !designName) {
        try {
          await createDesign('Official corridor', 'official')
          await handleOpen('Official corridor')
          setDesignList(await listDesigns())
        } catch {
          // another tab/StrictMode double-invoke already created it - fine
          await refreshList()
        }
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function refreshList() {
    setDesignList(await listDesigns())
  }

  async function handleCreate(template: 'blank' | 'official') {
    const name = newName.trim()
    if (!name) return
    setBusy('creating...')
    try {
      await createDesign(name, template)
      setNewName('')
      await refreshList()
      await handleOpen(name)
    } catch (e) {
      alert(String(e))
    } finally {
      setBusy(null)
    }
  }

  async function handleOpen(name: string) {
    if (!meta) return
    setBusy('loading design...')
    try {
      const [d, mat, delta] = await Promise.all([getDesign(name), getDesignMaterial(name), getDesignDemDelta(name)])
      openDesign(name, mat, delta, d.materials.materials, d.design.unlocked)
    } catch (e) {
      alert(String(e))
    } finally {
      setBusy(null)
    }
  }

  async function handleSave() {
    if (!designName) return
    setBusy('saving...')
    try {
      await saveDesign(designName)
      setDirty(false)
    } finally {
      setBusy(null)
    }
  }

  // Ctrl/Cmd+S saves the open design instead of triggering the browser's
  // "save page" dialog.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        if (designName) handleSave()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [designName])

  async function handleDeleteDesign() {
    if (!designName) return
    if (!confirm(`Delete design "${designName}"? This cannot be undone.`)) return
    setBusy('deleting...')
    try {
      await deleteDesign(designName)
      closeDesign()
      await refreshList()
    } finally {
      setBusy(null)
    }
  }

  async function handleLockToggle() {
    if (!designName) return
    const next = !unlocked
    setUnlockedLocal(next)
    await putLock(designName, next)
  }

  const activeMaterial = materials.find((m) => m.id === activeMaterialId) ?? materials[0]

  function updateActiveMaterial(patch: Partial<MaterialDef>) {
    if (!activeMaterial) return
    setMaterialsList(materials.map((m) => (m.id === activeMaterial.id ? { ...m, ...patch } : m)))
  }

  async function handleSaveMaterialProperties() {
    if (!designName) return
    setBusy('saving materials...')
    try {
      const clamped = await putMaterials(designName, materials)
      setMaterialsList(clamped)
    } finally {
      setBusy(null)
    }
  }

  async function handleNewMaterial() {
    if (!designName) return
    const label = prompt('Name for the new material:')
    if (!label) return
    const usedIds = materials.map((m) => m.id)
    const id = Math.max(99, ...usedIds) + 1
    const next: MaterialDef = {
      id,
      label,
      color: '#' + Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, '0'),
      infil_mmh: 100,
      manning_n: 0.05,
      depression_m: 0,
      builtin: false,
    }
    const updated = [...materials, next]
    setMaterialsList(updated)
    setActiveMaterialId(id)
    setBusy('saving materials...')
    try {
      const clamped = await putMaterials(designName, updated)
      setMaterialsList(clamped)
    } finally {
      setBusy(null)
    }
  }

  async function handleDeleteMaterial() {
    if (!designName || !activeMaterial || !meta || !material) return
    if (!confirm(`Delete "${activeMaterial.label}"? Every cell painted with it reverts to the base terrain.`)) return
    setBusy('removing material everywhere...')
    try {
      await deleteMaterialEverywhere(designName, material, meta.width, meta.height, activeMaterial.id)
      const remaining = materials.filter((m) => m.id !== activeMaterial.id)
      const clamped = await putMaterials(designName, remaining)
      setMaterialsList(clamped)
      if (clamped.length) setActiveMaterialId(clamped[0].id)
      bumpLayerVersion()
      setDirty(true)
    } finally {
      setBusy(null)
    }
  }

  return (
    <div
      style={{
        width: 300,
        flexShrink: 0,
        background: '#1a1a1a',
        color: '#ddd',
        borderLeft: '1px solid #333',
        overflowY: 'auto',
        font: '13px sans-serif',
        padding: 12,
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
      }}
    >
      {busy && <div style={{ color: '#facc15' }}>{busy}</div>}

      <section>
        <h3 style={h3}>Design</h3>
        {!designName ? (
          <>
            <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="new design name"
                style={{ flex: 1, ...inputStyle }}
              />
            </div>
            <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
              <button style={btn} onClick={() => handleCreate('official')}>
                New from official corridor
              </button>
            </div>
            <button style={btn} onClick={() => handleCreate('blank')}>
              New blank
            </button>
            <h4 style={h4}>Open existing</h4>
            {designList.length === 0 && <div style={{ opacity: 0.6 }}>no saved designs yet</div>}
            {designList.map((d) => (
              <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                <button style={{ ...btn, flex: 1, textAlign: 'left' }} onClick={() => handleOpen(d.name)}>
                  {d.name}
                </button>
              </div>
            ))}
          </>
        ) : (
          <>
            <div style={{ marginBottom: 8 }}>
              <strong>{designName}</strong>
              {dirty && <span style={{ color: '#facc15' }}> • unsaved changes</span>}
            </div>
            <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
              <button style={btn} onClick={handleSave} disabled={!dirty}>
                Save
              </button>
              <button style={btn} onClick={() => closeDesign()}>
                Close
              </button>
            </div>
            <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
              <button style={btn} onClick={undo ?? undefined} disabled={undoCount === 0}>
                Undo ({undoCount})
              </button>
              <button style={btn} onClick={redo ?? undefined} disabled={redoCount === 0}>
                Redo ({redoCount})
              </button>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8 }}>
              <input type="checkbox" checked={unlocked} onChange={handleLockToggle} />
              Unlock full domain (default: official zone only)
            </label>
            <button style={{ ...btn, marginTop: 8, color: '#f87171' }} onClick={handleDeleteDesign}>
              Delete design
            </button>
          </>
        )}
      </section>

      {designName && (
        <>
          <section>
            <h3 style={h3}>View</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              <button
                style={{ ...btn, background: overlayMode === 'materials' ? '#3b82f6' : btn.background }}
                onClick={() => setOverlayMode('materials')}
              >
                Materials
              </button>
              <button
                style={{ ...btn, background: overlayMode === 'elevation' ? '#3b82f6' : btn.background }}
                onClick={() => setOverlayMode('elevation')}
              >
                Elevation change
              </button>
            </div>
            {overlayMode === 'elevation' && (
              <div style={{ marginTop: 6, opacity: 0.8 }}>
                Heatmap of dem_delta: blue = dug, red = raised. Current range: ±{elevationRangeM.toFixed(2)} m.
              </div>
            )}
          </section>

          <section>
            <h3 style={h3}>Tool</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              {TOOLS.map((t) => (
                <button
                  key={t.id}
                  title={t.hint}
                  style={{ ...btn, background: tool === t.id ? '#3b82f6' : btn.background }}
                  onClick={() => setTool(t.id)}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div style={{ marginTop: 8 }}>{TOOLS.find((t) => t.id === tool)?.hint}</div>

            <label style={sliderLabel}>
              Brush radius: {brushRadiusM.toFixed(1)} m
              <input
                type="range"
                min={0.5}
                max={30}
                step={0.5}
                value={brushRadiusM}
                onChange={(e) => setBrushRadiusM(Number(e.target.value))}
              />
            </label>
            <label style={sliderLabel}>
              Sculpt strength: {brushStrengthMps.toFixed(2)} m/s (hold to build up)
              <input
                type="range"
                min={0.05}
                max={0.5}
                step={0.01}
                value={brushStrengthMps}
                onChange={(e) => setBrushStrengthMps(Number(e.target.value))}
              />
            </label>
            <label style={sliderLabel}>
              Brush softness: {brushSoftness.toFixed(2)}
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={brushSoftness}
                onChange={(e) => setBrushSoftness(Number(e.target.value))}
              />
            </label>
          </section>

          <section>
            <h3 style={h3}>Materials</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 4, marginBottom: 8 }}>
              {materials.map((m) => (
                <button
                  key={m.id}
                  title={materialTooltip(m)}
                  onClick={() => setActiveMaterialId(m.id)}
                  style={{
                    height: 32,
                    background: m.color,
                    border: activeMaterialId === m.id ? '2px solid #fff' : '1px solid #444',
                    borderRadius: 4,
                    cursor: 'pointer',
                  }}
                />
              ))}
            </div>
            <button style={{ ...btn, width: '100%', marginBottom: 8 }} onClick={handleNewMaterial}>
              + New material
            </button>

            {activeMaterial && (
              <div style={{ border: '1px solid #333', borderRadius: 4, padding: 8 }}>
                <div style={{ marginBottom: 4, fontWeight: 'bold' }}>{activeMaterial.label}</div>
                <div style={{ marginBottom: 8, opacity: 0.8 }}>{materialTooltip(activeMaterial)}</div>
                <label style={sliderLabel}>
                  Color
                  <input
                    type="color"
                    value={activeMaterial.color}
                    onChange={(e) => updateActiveMaterial({ color: e.target.value })}
                  />
                </label>
                <label style={sliderLabel}>
                  Infiltration: {activeMaterial.infil_mmh.toFixed(0)} mm/h (how fast the ground drinks water)
                  <input
                    type="range"
                    min={0}
                    max={1000}
                    step={5}
                    value={activeMaterial.infil_mmh}
                    onChange={(e) => updateActiveMaterial({ infil_mmh: Number(e.target.value) })}
                  />
                </label>
                <label style={sliderLabel}>
                  Roughness: {activeMaterial.manning_n.toFixed(3)} (higher = slows water down more)
                  <input
                    type="range"
                    min={0.01}
                    max={0.5}
                    step={0.005}
                    value={activeMaterial.manning_n}
                    onChange={(e) => updateActiveMaterial({ manning_n: Number(e.target.value) })}
                  />
                </label>
                <label style={sliderLabel}>
                  Detention depth: {activeMaterial.depression_m.toFixed(2)} m (how deep a hollow it forms)
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={activeMaterial.depression_m}
                    onChange={(e) => updateActiveMaterial({ depression_m: Number(e.target.value) })}
                  />
                </label>
                <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
                  <button style={btn} onClick={handleSaveMaterialProperties}>
                    Save properties
                  </button>
                  <button style={{ ...btn, color: '#f87171' }} onClick={handleDeleteMaterial}>
                    Delete
                  </button>
                </div>
              </div>
            )}
          </section>

          <RunPanel />
        </>
      )}
    </div>
  )
}

