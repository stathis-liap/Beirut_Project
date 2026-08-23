import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { useStore, type MaterialDef } from './store'
import { computeEffectiveZ, computeNodataOverlay, fillInvalidNearest, materialDepressionTable } from './terrain'

/** The corridor's demolition at one cell, if the open design applies it.
 * Kept out of demDelta so the per-cell sculpt clamp can't claw it back;
 * every place that computes a rendered ground height has to add it. */
function clearAt(st: { clearDelta: Float32Array | null; clearsBuildings: boolean }, idx: number) {
  return st.clearsBuildings && st.clearDelta ? st.clearDelta[idx] : 0
}
import { buildEditableMask, INITIAL_TAP_DT, materialColorTable, maxAbsDelta, renderMaterialRect, Stroke } from './editor'
import { patchDesign } from './api'

const PERF_STRIDE = 2 // "performance mode" decimation factor for weaker machines
// THREE.Raycaster does a naive linear scan over every triangle - at full
// detail (~5.4M triangles) that makes each raycast take so long it starves
// the interaction, silently throttling how much a stroke can accumulate per
// second of real time. Raycasting instead against a separate, coarse,
// never-rendered "picking" mesh keeps hit-testing effectively instant
// regardless of the display mesh's resolution.
const PICK_STRIDE = 12

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

interface Rect {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

interface GridMesh {
  gw: number
  gh: number
  positions: Float32Array
  uvs: Float32Array
  indices: Uint32Array
}

/** Builds a heightfield grid at the given stride: vertex (r,c) sits at
 * (sc*res, effectiveZ, sr*res) where sc=min(c*stride,width-1) etc. `z` must
 * already be filled across no-survey-data cells (see `fillInvalidNearest`)
 * so every quad is buildable - no dropped quads means no holes to fall
 * through and no fixed-height fill plane to create a stretched cliff. */
function buildGridMesh(width: number, height: number, res: number, z: Float32Array, stride: number): GridMesh {
  const gw = Math.ceil(width / stride)
  const gh = Math.ceil(height / stride)
  const positions = new Float32Array(gw * gh * 3)
  const uvs = new Float32Array(gw * gh * 2)
  for (let r = 0; r < gh; r++) {
    const sr = Math.min(r * stride, height - 1)
    for (let c = 0; c < gw; c++) {
      const sc = Math.min(c * stride, width - 1)
      const idx = sr * width + sc
      const vi = (r * gw + c) * 3
      positions[vi] = sc * res
      positions[vi + 1] = z[idx]
      positions[vi + 2] = sr * res
      const ui = (r * gw + c) * 2
      uvs[ui] = sc / (width - 1)
      uvs[ui + 1] = 1 - sr / (height - 1)
    }
  }
  const indexList: number[] = []
  for (let r = 0; r < gh - 1; r++) {
    for (let c = 0; c < gw - 1; c++) {
      const a = r * gw + c
      const b = a + 1
      const cc = a + gw
      const d = cc + 1
      indexList.push(a, cc, b, b, cc, d)
    }
  }
  return { gw, gh, positions, uvs, indices: new Uint32Array(indexList) }
}

/** Per-vertex normal via the standard heightfield central-difference formula
 * (cross of the row-tangent and column-tangent through each vertex's
 * immediate neighbours), restricted to [r0,r1]x[c0,c1]. Visually equivalent
 * to averaging adjacent face normals but O(1) per vertex instead of O(faces
 * per vertex), so a brush stroke's shading update costs only the touched
 * patch rather than the whole mesh. */
function updateLocalNormals(posArr: Float32Array, normArr: Float32Array, gw: number, gh: number, r0: number, r1: number, c0: number, c1: number) {
  for (let r = r0; r <= r1; r++) {
    const rU = Math.max(r - 1, 0)
    const rD = Math.min(r + 1, gh - 1)
    for (let c = c0; c <= c1; c++) {
      const cL = Math.max(c - 1, 0)
      const cR = Math.min(c + 1, gw - 1)
      const iL = (r * gw + cL) * 3
      const iR = (r * gw + cR) * 3
      const iU = (rU * gw + c) * 3
      const iD = (rD * gw + c) * 3
      const txx = posArr[iR] - posArr[iL]
      const txy = posArr[iR + 1] - posArr[iL + 1]
      const txz = posArr[iR + 2] - posArr[iL + 2]
      const tzx = posArr[iD] - posArr[iU]
      const tzy = posArr[iD + 1] - posArr[iU + 1]
      const tzz = posArr[iD + 2] - posArr[iU + 2]
      const nx = tzy * txz - tzz * txy
      const ny = tzz * txx - tzx * txz
      const nz = tzx * txy - tzy * txx
      const len = Math.hypot(nx, ny, nz) || 1
      const vi = (r * gw + c) * 3
      normArr[vi] = nx / len
      normArr[vi + 1] = ny / len
      normArr[vi + 2] = nz / len
    }
  }
}

export default function Scene3D() {
  const containerRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const meshRef = useRef<THREE.Mesh | null>(null)
  // a coarse, never-rendered mesh used only for raycasting - see PICK_STRIDE.
  const pickingMeshRef = useRef<THREE.Mesh | null>(null)
  const rafRef = useRef<number | null>(null)
  const cameraFramed = useRef(false)

  // live-brushing support
  const gridRef = useRef<{ stride: number; gw: number; gh: number } | null>(null)
  const pickGridRef = useRef<{ stride: number; gw: number; gh: number } | null>(null)
  const orthoImageRef = useRef<HTMLImageElement | null>(null)
  const orthoCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const textureCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const textureRef = useRef<THREE.CanvasTexture | null>(null)
  const editableMaskRef = useRef<Uint8Array | null>(null)
  const strokeRef = useRef<Stroke | null>(null)
  const pointerCell = useRef<{ row: number; col: number } | null>(null)
  const lastStampTime = useRef(0)
  const cursorRingRef = useRef<THREE.Mesh | null>(null)
  const raycasterRef = useRef(new THREE.Raycaster())
  // materialDepressionTable() allocates a fresh Map from the materials array
  // every call; during a live drag getYAt()/updateLiveRegion() call it every
  // frame (or on every pointermove), which was pure GC churn for a table
  // that only actually changes when the material list itself changes.
  const depressionCacheRef = useRef<{ materials: MaterialDef[] | null; table: Map<number, number> }>({
    materials: null,
    table: new Map(),
  })
  function getDepressionTable(materials: MaterialDef[]): Map<number, number> {
    const cache = depressionCacheRef.current
    if (cache.materials !== materials) {
      cache.table = materialDepressionTable(materials)
      cache.materials = materials
    }
    return cache.table
  }

  const meta = useStore((s) => s.meta)
  const dem = useStore((s) => s.dem)
  const masksRgba = useStore((s) => s.masksRgba)
  const designName = useStore((s) => s.designName)
  const material = useStore((s) => s.material)
  const demDelta = useStore((s) => s.demDelta)
  const clearDelta = useStore((s) => s.clearDelta)
  const clearsBuildings = useStore((s) => s.clearsBuildings)
  const materials = useStore((s) => s.materials)
  const unlocked = useStore((s) => s.unlocked)
  const editVersion = useStore((s) => s.editVersion)
  const setDirty = useStore((s) => s.setDirty)
  const setUndoRedoCounts = useStore((s) => s.setUndoRedoCounts)
  const bumpLayerVersion = useStore((s) => s.bumpLayerVersion)
  const setElevationRangeM = useStore((s) => s.setElevationRangeM)

  const [exaggeration, setExaggeration] = useState(1.5)
  // full resolution by default so small brush strokes are actually visible;
  // "performance mode" is the opt-in for weaker machines.
  const [fullDetail, setFullDetail] = useState(true)
  const [status, setStatus] = useState('loading terrain...')
  const [sculptWarning, setSculptWarning] = useState<string | null>(null)
  const exaggerationRef = useRef(exaggeration)
  exaggerationRef.current = exaggeration

  // ---- editable mask, mirrors MapView's ----
  useEffect(() => {
    if (!meta || !masksRgba) return
    editableMaskRef.current = buildEditableMask(masksRgba, meta.width, meta.height, unlocked)
  }, [meta, masksRgba, unlocked])

  // ---- one-time renderer/scene/camera/controls/brush-input setup ----
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x141414)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(55, 1, 0.5, 50000)
    cameraRef.current = camera

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    // Left button is reserved for the sculpt/paint brush; navigate with
    // right (orbit) and middle (pan), scroll to zoom - consistent with the
    // 2D view's "left acts, other buttons navigate" convention.
    controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE }
    controlsRef.current = controls

    scene.add(new THREE.AmbientLight(0xffffff, 0.7))
    const sun = new THREE.DirectionalLight(0xffffff, 0.9)
    sun.position.set(1, 1.6, 0.6)
    scene.add(sun)

    const ringGeom = new THREE.RingGeometry(0.85, 1.0, 32)
    // depthTest stays on: the ring must be properly occluded by nearer
    // geometry (e.g. a building between the camera and the marked point),
    // otherwise it can appear to float over the wrong surface and look
    // misaligned with where the brush is actually landing.
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x4ade80, side: THREE.DoubleSide, transparent: true, opacity: 0.9 })
    const ring = new THREE.Mesh(ringGeom, ringMat)
    ring.rotation.x = -Math.PI / 2
    ring.visible = false
    ring.renderOrder = 999
    scene.add(ring)
    cursorRingRef.current = ring

    const resize = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      if (w === 0 || h === 0) return
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)

    function tick() {
      controls.update()
      renderer.render(scene, camera)
      rafRef.current = requestAnimationFrame(tick)
    }
    tick()

    // ---- brush raycasting against the live mesh ----
    const dom = renderer.domElement

    function cellFromEvent(e: PointerEvent): { row: number; col: number } | null {
      if (!pickingMeshRef.current || !cameraRef.current) return null
      const rect = dom.getBoundingClientRect()
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1)
      raycasterRef.current.setFromCamera(ndc, cameraRef.current)
      const hits = raycasterRef.current.intersectObject(pickingMeshRef.current, false)
      if (!hits.length) return null
      const res = useStore.getState().meta!.res
      return { row: hits[0].point.z / res, col: hits[0].point.x / res }
    }

    function updateLiveRegion(rect: Rect) {
      const st = useStore.getState()
      if (!meshRef.current || !gridRef.current || !st.meta || !st.dem || !st.material || !st.demDelta || !st.masksRgba) return
      const { stride, gw, gh } = gridRef.current
      const { width, height } = st.meta
      const depression = getDepressionTable(st.materials)
      const pos = meshRef.current.geometry.attributes.position as THREE.BufferAttribute
      const arr = pos.array as Float32Array

      const r0 = Math.max(0, Math.floor(rect.minY / stride) - 1)
      const r1 = Math.min(Math.ceil(height / stride) - 1, Math.ceil(rect.maxY / stride) + 1)
      const c0 = Math.max(0, Math.floor(rect.minX / stride) - 1)
      const c1 = Math.min(gw - 1, Math.ceil(rect.maxX / stride) + 1)
      for (let r = r0; r <= r1; r++) {
        const sr = Math.min(r * stride, height - 1)
        for (let c = c0; c <= c1; c++) {
          const sc = Math.min(c * stride, width - 1)
          const idx = sr * width + sc
          if (st.masksRgba[idx * 4 + 3] <= 127) continue // leave nodata fill Z as-is
          const cls = st.material[idx]
          const dep = cls === 0 ? 0 : (depression.get(cls) ?? 0)
          const vi = (r * gw + c) * 3
          arr[vi + 1] = st.dem[idx] - dep + st.demDelta[idx] + clearAt(st, idx)
        }
      }
      pos.needsUpdate = true
      // Recompute normals only for the touched patch (plus a 1-vertex halo
      // for correct shading at its edge) instead of geometry.computeVertexNormals(),
      // which walks every one of the ~5.4M triangles in the full-detail mesh
      // regardless of how small the edit was - that full-mesh scan, even
      // throttled, was the main source of hitching while sculpting. Uses the
      // standard heightfield central-difference normal instead of the
      // per-face-averaged one; visually equivalent, O(touched cells) instead
      // of O(triangles). A single real computeVertexNormals() runs once on
      // stroke finish to guarantee it exactly matches a full rebuild.
      const norm = meshRef.current.geometry.attributes.normal as THREE.BufferAttribute
      updateLocalNormals(arr, norm.array as Float32Array, gw, gh, r0, r1, c0, c1)
      norm.needsUpdate = true

      // keep the (coarse) picking mesh's heights in sync too, so raycasting
      // during an active stroke reflects the ground the user is actively
      // reshaping rather than its pre-stroke height. Cheap: this mesh is
      // never rendered, so there's no GPU upload to trigger.
      if (pickingMeshRef.current && pickGridRef.current) {
        const { stride: pstride, gw: pgw } = pickGridRef.current
        const parr = (pickingMeshRef.current.geometry.attributes.position as THREE.BufferAttribute).array as Float32Array
        const pr0 = Math.max(0, Math.floor(rect.minY / pstride) - 1)
        const pr1 = Math.min(Math.ceil(height / pstride) - 1, Math.ceil(rect.maxY / pstride) + 1)
        const pc0 = Math.max(0, Math.floor(rect.minX / pstride) - 1)
        const pc1 = Math.min(pgw - 1, Math.ceil(rect.maxX / pstride) + 1)
        for (let r = pr0; r <= pr1; r++) {
          const sr = Math.min(r * pstride, height - 1)
          for (let c = pc0; c <= pc1; c++) {
            const sc = Math.min(c * pstride, width - 1)
            const idx = sr * width + sc
            if (st.masksRgba[idx * 4 + 3] <= 127) continue
            const cls = st.material[idx]
            const dep = cls === 0 ? 0 : (depression.get(cls) ?? 0)
            const vi = (r * pgw + c) * 3
            parr[vi + 1] = st.dem[idx] - dep + st.demDelta[idx] + clearAt(st, idx)
          }
        }
      }

      // repaint just this rect of the texture: base ortho pixels, then the
      // material tint on top (mirrors the full-rebuild compositing below).
      if (orthoCanvasRef.current && textureCanvasRef.current && textureRef.current) {
        const w = rect.maxX - rect.minX + 1
        const h = rect.maxY - rect.minY + 1
        const ctx = textureCanvasRef.current.getContext('2d')!
        ctx.drawImage(orthoCanvasRef.current, rect.minX, rect.minY, w, h, rect.minX, rect.minY, w, h)
        if (st.materials.length > 0) {
          const colors = materialColorTable(st.materials)
          const overlay = renderMaterialRect(st.material, width, colors, rect)
          const tmp = document.createElement('canvas')
          tmp.width = w
          tmp.height = h
          tmp.getContext('2d')!.putImageData(overlay, 0, 0)
          ctx.drawImage(tmp, rect.minX, rect.minY)
        }
        textureRef.current.needsUpdate = true
      }
    }

    function updateCursorRing(row: number, col: number) {
      const st = useStore.getState()
      if (!st.meta || !cursorRingRef.current) return
      const flRow = Math.floor(row)
      const flCol = Math.floor(col)
      if (flRow < 0 || flRow >= st.meta.height || flCol < 0 || flCol >= st.meta.width) {
        cursorRingRef.current.visible = false
        return
      }
      const editable = editableMaskRef.current ? editableMaskRef.current[flRow * st.meta.width + flCol] === 1 : false
      const res = st.meta.res
      const radiusM = st.brushRadiusM
      // the ring is a plain scene child (not parented to the mesh), so it
      // must apply the same vertical exaggeration as mesh.scale.y itself to
      // sit on the visible surface rather than floating above/below it.
      const yWorld = (meshRef.current ? getYAt(row, col) : 0) * exaggerationRef.current + 0.2
      cursorRingRef.current.position.set(col * res, yWorld, row * res)
      cursorRingRef.current.scale.set(radiusM, radiusM, 1)
      ;(cursorRingRef.current.material as THREE.MeshBasicMaterial).color.set(editable ? 0x4ade80 : 0xf87171)
      cursorRingRef.current.visible = !!st.designName
    }

    function getYAt(row: number, col: number): number {
      const st = useStore.getState()
      if (!st.dem || !st.meta) return 0
      const r = Math.max(0, Math.min(st.meta.height - 1, Math.round(row)))
      const c = Math.max(0, Math.min(st.meta.width - 1, Math.round(col)))
      const idx = r * st.meta.width + c
      const depression = getDepressionTable(st.materials)
      const mat = st.material ?? new Uint16Array(0)
      const cls = mat.length ? mat[idx] : 0
      const dep = cls === 0 ? 0 : (depression.get(cls) ?? 0)
      return st.dem[idx] - dep + (st.demDelta ? st.demDelta[idx] : 0) + clearAt(st, idx)
    }

    // Keeps stamping at the last raycast cell every frame while a stroke is
    // active, so sculpt tools (raise/lower/flatten/smooth) accumulate over
    // time even when the pointer isn't moving ("m/s of hold") - mirrors
    // MapView's 2D brush loop.
    let brushTickId: number | null = null
    function brushTick() {
      if (!strokeRef.current || !pointerCell.current) {
        brushTickId = null
        return
      }
      const now = performance.now()
      const dt = Math.min((now - lastStampTime.current) / 1000, 0.1)
      lastStampTime.current = now
      const rect = strokeRef.current.stampAt(pointerCell.current.row, pointerCell.current.col, dt)
      if (rect) updateLiveRegion(rect)
      brushTickId = requestAnimationFrame(brushTick)
    }

    function onPointerDown(e: PointerEvent) {
      if (e.button !== 0) return
      const st = useStore.getState()
      if (!st.designName || !st.material || !st.demDelta || !st.dem || !st.meta || !editableMaskRef.current) return
      const cell = cellFromEvent(e)
      if (!cell) return
      e.preventDefault()
      controls.enabled = false
      const radiusCells = st.brushRadiusM / st.meta.res
      strokeRef.current = new Stroke(
        st.tool,
        st.activeMaterialId,
        st.meta.width,
        st.meta.height,
        st.dem,
        st.material,
        st.demDelta,
        editableMaskRef.current,
        radiusCells,
        st.brushStrengthMps,
        st.brushSoftness,
        st.clearsBuildings ? st.clearDelta : null,
      )
      lastStampTime.current = performance.now()
      pointerCell.current = cell
      const rect = strokeRef.current.stampAt(cell.row, cell.col, INITIAL_TAP_DT)
      if (rect) updateLiveRegion(rect)
      dom.setPointerCapture(e.pointerId)
      if (brushTickId === null) brushTickId = requestAnimationFrame(brushTick)
    }

    function onPointerMove(e: PointerEvent) {
      const cell = cellFromEvent(e)
      if (cell) updateCursorRing(cell.row, cell.col)
      else if (cursorRingRef.current) cursorRingRef.current.visible = false

      if (!strokeRef.current) return
      pointerCell.current = cell
      if (!cell) return
      const now = performance.now()
      const dt = Math.min((now - lastStampTime.current) / 1000, 0.1)
      lastStampTime.current = now
      const rect = strokeRef.current.moveTo(cell.row, cell.col, dt)
      if (rect) updateLiveRegion(rect)
    }

    function onPointerUp(e: PointerEvent) {
      if (!strokeRef.current) return
      controls.enabled = true
      if (brushTickId !== null) {
        cancelAnimationFrame(brushTickId)
        brushTickId = null
      }
      pointerCell.current = null
      const rec = strokeRef.current.finish()
      strokeRef.current = null
      dom.releasePointerCapture(e.pointerId)
      // one real full-mesh normal recompute now that the stroke is over, so
      // the per-stamp local approximation never has a chance to drift from
      // what a full rebuild would produce (cheap here: once per stroke, not
      // once per frame).
      if (rec && meshRef.current) meshRef.current.geometry.computeVertexNormals()
      if (!rec) return

      const st = useStore.getState()
      const mgr = st.undoManager
      mgr?.push(rec)
      if (mgr) {
        const counts = mgr.counts()
        setUndoRedoCounts(counts.undo, counts.redo)
      }
      setDirty(true)

      if (rec.beforeDemDelta && rec.afterDemDelta) {
        if (st.demDelta) setElevationRangeM(maxAbsDelta(st.demDelta))
        let maxChange = 0
        for (let i = 0; i < rec.afterDemDelta.length; i++) {
          const d = Math.abs(rec.afterDemDelta[i] - rec.beforeDemDelta[i])
          if (d > maxChange) maxChange = d
        }
        if (maxChange > 2.0) {
          setSculptWarning(`That stroke moved the ground up to ${maxChange.toFixed(1)} m in one go — double check this is intended.`)
          setTimeout(() => setSculptWarning(null), 6000)
        }
      }

      // keep the 2D view and the shared undo history fully in sync (also
      // triggers this component's own full rebuild via editVersion).
      bumpLayerVersion()

      if (st.designName) {
        patchDesign(st.designName, { x: rec.x, y: rec.y, w: rec.w, h: rec.h, material: rec.afterMaterial, demDelta: rec.afterDemDelta })
      }
    }

    dom.addEventListener('pointerdown', onPointerDown)
    dom.addEventListener('pointermove', onPointerMove)
    dom.addEventListener('pointerup', onPointerUp)
    dom.addEventListener('pointerleave', () => {
      if (cursorRingRef.current) cursorRingRef.current.visible = false
    })

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      if (brushTickId !== null) cancelAnimationFrame(brushTickId)
      ro.disconnect()
      dom.removeEventListener('pointerdown', onPointerDown)
      dom.removeEventListener('pointermove', onPointerMove)
      dom.removeEventListener('pointerup', onPointerUp)
      controls.dispose()
      renderer.dispose()
      if (renderer.domElement.parentNode === container) container.removeChild(renderer.domElement)
      if (meshRef.current) {
        meshRef.current.geometry.dispose()
        ;(meshRef.current.material as THREE.Material).dispose()
      }
      if (pickingMeshRef.current) pickingMeshRef.current.geometry.dispose()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- apply vertical exaggeration without a full rebuild ----
  useEffect(() => {
    if (meshRef.current) meshRef.current.scale.y = exaggeration
  }, [exaggeration])

  // ---- left button: orbit when just exploring, brush once a design is open ----
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.mouseButtons.LEFT = designName ? null : THREE.MOUSE.ROTATE
    }
  }, [designName])

  // ---- load the (static) ortho image once ----
  useEffect(() => {
    loadImage('/api/terrain/ortho.png').then((img) => {
      orthoImageRef.current = img
    })
  }, [])

  // ---- (re)build the heightmap mesh + texture ----
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene || !meta || !dem || !masksRgba) return

    let disposed = false
    setStatus('building mesh...')

    const stride = fullDetail ? 1 : PERF_STRIDE
    const gw = Math.ceil(meta.width / stride)
    const gh = Math.ceil(meta.height / stride)
    const res = meta.res
    gridRef.current = { stride, gw, gh }

    // a design may not be open yet: fall back to zero material/delta so the
    // bare terrain still renders.
    const mat = material ?? new Uint16Array(meta.width * meta.height)
    const delta = demDelta ?? new Float32Array(meta.width * meta.height)
    const depression = getDepressionTable(materials)
    const z = computeEffectiveZ(dem, delta, mat, depression,
                                clearsBuildings ? clearDelta : null)

    let validMin = Infinity
    for (let i = 0; i < dem.length; i++) {
      if (masksRgba[i * 4 + 3] > 127 && z[i] < validMin) validMin = z[i]
    }
    if (!isFinite(validMin)) validMin = 0

    // No-survey-data cells get the elevation of their nearest valid cell
    // instead of either (a) a dropped quad, which left a real hole you could
    // fly the camera through, or (b) a single flat fill height, which built
    // a stretched cliff wherever the real terrain was far from that height.
    // Nearest-neighbour extrapolation keeps the surface continuous with
    // whatever terrain actually borders the gap.
    const zFilled = fillInvalidNearest(z, masksRgba, meta.width, meta.height)

    const display = buildGridMesh(meta.width, meta.height, res, zFilled, stride)
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(display.positions, 3))
    geo.setAttribute('uv', new THREE.BufferAttribute(display.uvs, 2))
    geo.setIndex(new THREE.BufferAttribute(display.indices, 1))
    geo.computeVertexNormals()

    // separate, coarse, never-rendered geometry used only for raycasting -
    // see PICK_STRIDE.
    const pick = buildGridMesh(meta.width, meta.height, res, zFilled, PICK_STRIDE)
    pickGridRef.current = { stride: PICK_STRIDE, gw: pick.gw, gh: pick.gh }
    if (pickingMeshRef.current) pickingMeshRef.current.geometry.dispose()
    const pickGeo = new THREE.BufferGeometry()
    pickGeo.setAttribute('position', new THREE.BufferAttribute(pick.positions, 3))
    pickGeo.setIndex(new THREE.BufferAttribute(pick.indices, 1))
    pickingMeshRef.current = new THREE.Mesh(pickGeo) // intentionally never added to the scene

    setStatus('loading texture...')
    ;(orthoImageRef.current ? Promise.resolve(orthoImageRef.current) : loadImage('/api/terrain/ortho.png'))
      .then((img) => {
        if (disposed) return
        orthoImageRef.current = img

        const orthoCanvas = orthoCanvasRef.current ?? document.createElement('canvas')
        orthoCanvas.width = meta.width
        orthoCanvas.height = meta.height
        orthoCanvas.getContext('2d')!.drawImage(img, 0, 0)
        orthoCanvasRef.current = orthoCanvas

        const canvas = textureCanvasRef.current ?? document.createElement('canvas')
        canvas.width = meta.width
        canvas.height = meta.height
        const ctx = canvas.getContext('2d')!
        ctx.drawImage(orthoCanvas, 0, 0)

        if (materials.length > 0) {
          const colors = materialColorTable(materials)
          const full = { minX: 0, minY: 0, maxX: meta.width - 1, maxY: meta.height - 1 }
          const overlay = renderMaterialRect(mat, meta.width, colors, full)
          const overlayCanvas = document.createElement('canvas')
          overlayCanvas.width = meta.width
          overlayCanvas.height = meta.height
          overlayCanvas.getContext('2d')!.putImageData(overlay, 0, 0)
          ctx.drawImage(overlayCanvas, 0, 0)
        }

        // the ortho export bakes literal black pixels into the no-survey-data
        // area; now that that area has real (extrapolated) geometry instead
        // of a hole, tint it to read as "no data" instead of showing those
        // raw black pixels as if they were photographed ground - mirrors the
        // 2D view's nodata overlay exactly.
        const nodata = computeNodataOverlay(masksRgba, meta.width, meta.height)
        const nodataCanvas = document.createElement('canvas')
        nodataCanvas.width = meta.width
        nodataCanvas.height = meta.height
        nodataCanvas.getContext('2d')!.putImageData(new ImageData(nodata, meta.width, meta.height), 0, 0)
        ctx.drawImage(nodataCanvas, 0, 0)

        textureCanvasRef.current = canvas

        const texture = textureRef.current ?? new THREE.CanvasTexture(canvas)
        texture.image = canvas
        texture.colorSpace = THREE.SRGBColorSpace
        texture.needsUpdate = true
        textureRef.current = texture

        if (meshRef.current) {
          scene.remove(meshRef.current)
          meshRef.current.geometry.dispose()
          ;(meshRef.current.material as THREE.Material).dispose()
        }
        const meshMat = new THREE.MeshStandardMaterial({ map: texture, side: THREE.DoubleSide })
        const mesh = new THREE.Mesh(geo, meshMat)
        mesh.scale.y = exaggerationRef.current
        scene.add(mesh)
        meshRef.current = mesh

        // Read the height the mesh ACTUALLY renders at a grid cell, straight
        // off the geometry buffer. Every "is the terrain really fixed?" round
        // trip so far has been argued from files and API responses, which
        // cannot distinguish a stale browser or a rendering bug from bad data;
        // this can. Kept deliberately: it costs nothing and closes that loop.
        //   __scene3dDebug.getY(row, col)  -> metres, or null if not built yet
        ;(window as unknown as Record<string, unknown>).__scene3dDebug = {
          getY(row: number, col: number) {
            const g = meshRef.current?.geometry
            const grid = gridRef.current
            if (!g || !grid) return null
            const { stride, gw, gh } = grid
            const r = Math.min(Math.round(row / stride), gh - 1)
            const c = Math.min(Math.round(col / stride), gw - 1)
            return (g.attributes.position.array as Float32Array)[(r * gw + c) * 3 + 1]
          },
        }

        if (!cameraFramed.current && cameraRef.current && controlsRef.current) {
          const cx = (meta.width * res) / 2
          const cz = (meta.height * res) / 2
          controlsRef.current.target.set(cx, validMin, cz)
          cameraRef.current.position.set(cx - meta.width * res * 0.35, validMin + meta.height * res * 0.45, cz + meta.height * res * 0.55)
          controlsRef.current.update()
          cameraFramed.current = true
        }
        setStatus('')
      })
      .catch((e) => setStatus('error: ' + e))

    return () => {
      disposed = true
    }
  }, [meta, dem, masksRgba, material, demDelta, clearDelta, clearsBuildings,
      materials, editVersion, fullDetail])

  return (
    <div style={{ position: 'relative', height: '100%' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      <div
        style={{
          position: 'absolute',
          right: 12,
          top: 12,
          background: 'rgba(0,0,0,0.6)',
          color: '#fff',
          font: '12px sans-serif',
          padding: '8px 10px',
          borderRadius: 4,
          width: 200,
        }}
      >
        <label style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          Vertical exaggeration: {exaggeration.toFixed(1)}x
          <input type="range" min={1} max={3} step={0.1} value={exaggeration} onChange={(e) => setExaggeration(Number(e.target.value))} />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8 }}>
          <input type="checkbox" checked={fullDetail} onChange={(e) => setFullDetail(e.target.checked)} />
          Full detail (slower)
        </label>
        {!designName && <div style={{ marginTop: 8, opacity: 0.7 }}>No design open — showing base terrain only.</div>}
        {designName && (
          <div style={{ marginTop: 8, opacity: 0.7 }}>Left-drag to paint/sculpt · right-drag to orbit · middle-drag to pan</div>
        )}
      </div>
      {status && <div style={{ position: 'absolute', left: 12, top: 12, color: '#fff', font: '13px sans-serif' }}>{status}</div>}
      {sculptWarning && (
        <div
          style={{
            position: 'absolute',
            left: '50%',
            top: 12,
            transform: 'translateX(-50%)',
            background: 'rgba(120,53,15,0.9)',
            color: '#fff',
            font: '13px sans-serif',
            padding: '8px 14px',
            borderRadius: 4,
            border: '1px solid #f59e0b',
          }}
        >
          ⚠ {sculptWarning}
        </div>
      )}
    </div>
  )
}
