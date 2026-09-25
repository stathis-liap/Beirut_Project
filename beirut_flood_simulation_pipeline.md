# Hybrid 2D/3D Urban Flood Simulation Pipeline

This document outlines the schematic architecture for a hybrid urban flood simulation system. It utilizes a lightweight 2D macro-simulation globally and triggers a computationally intensive 3D particle simulation locally on demand.

## Phase 1: Data Ingestion & Geometry Repair (`DATA/FILE_NAME.LAS` → Watertight Terrain)
Raw point clouds inherently have occlusion holes (e.g., behind walls or under dense trees). If left unpatched, 3D fluid particles will "leak" through the floor, draining system resources and invalidating the physical volume of the water.
*   **Action:** Project the `.las` points onto a grid.
*   **Repair:** Apply an interpolation algorithm (like Delaunay triangulation or Inverse Distance Weighting) across the empty spaces to patch the holes, creating a continuous, seamless bare-earth surface.

## Phase 2: The Macro Layer (Global 2D Simulation)
Run the Shallow Water Equation (SWE) arithmetic simulation across the repaired, continuous 2D grid.
*   **Action:** This layer calculates general water depth and momentum city-wide. It renders top-down in an orthographic view, acting as the real-time visual baseline and calculating the broader physics.

## Phase 3: The Event Listener (Interactive Trigger)
The application interface listens for user input on the 2D orthographic map.
*   **Action:** Upon `CTRL+CLICK`, the system captures the 2D world coordinates (x, y) and defines a local, predefined bounding box (e.g., 50m × 50m) around the cursor.

## Phase 4: Generating the Micro-Environment (Localized 3D Reconstruction)
The system queries the original `.las` dataset, fetching only the points strictly within the new bounding box.
*   **Action:** Apply a meshing algorithm (like Poisson Surface Reconstruction) to these local points. This converts the point cloud into a watertight 3D polygon mesh with collision physics enabled, ensuring particles will physically interact with the buildings and streets instead of falling into a void.

## Phase 5: Calculating the Math (Particle Emitter Initialization)
The localized physics engine calculates the exact number of fluid particles required to simulate the rain event within that specific bounding box.
*   **The Math:** The total number of particles ($N$) is derived from the target surface area ($A$), the rainfall rate ($R$), the duration of the rain event ($t$), and the pre-defined volume for a single particle ($V_p$).

$$N = \frac{A \cdot R \cdot t}{V_p}$$

*(Note: Units must be aligned before running this calculation—convert $R$ from mm/h to meters so it matches the spatial units of $A$ and $V_p$).*

## Phase 6: Simulating the Micro-Flow (3D Particle Fluid Dynamics)
The engine spawns the $N$ particles above the 3D reconstructed mesh.
*   **Action:** Using a particle-based fluid method (like Smoothed Particle Hydrodynamics - SPH), the engine calculates gravity, collision, and viscosity. The user watches a high-fidelity simulation of water splashing against curbs and flowing around obstacles at their exact chosen location.
