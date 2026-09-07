# Connectivity-Aware Grasp Exploration

Experimental framework for studying whether the **connectivity structure of robotic grasp spaces in SE(3)** can improve exploration and subsequent grasp discovery.

The experiments use ACRONYM grasp candidates and compare connectivity-aware sampling with random and geometry-based baselines.

## Project Structure

```text
src/
├── ablations/
├── e1/
│   ├── convergence.py
│   ├── convergence_plot.py
│   └── vis.py
├── e2/
│   ├── __init__.py
│   └── experiment.py
├── e3/
│   ├── analyze_meta.py
│   ├── experiment.py
│   ├── plot_results.py
│   ├── stats_exploration.py
│   ├── stats_failure.py
│   └── stats_final.py
├── e4/
└── utils/
    ├── connectivity.py
    ├── geometry.py
    ├── grasp_io.py
    └── samplers.py
```

Legacy and exploratory scripts are omitted from this overview.

## Method

A grasp is represented as a rigid transformation \(T \in SE(3)\). Pairwise distance combines normalized translational and rotational distances:

$$
d(T_i,T_j)
=
\sqrt{
\left(\frac{d_p}{s_p}\right)^2+
\left(\frac{d_R}{s_R}\right)^2
}.
$$

At each distance threshold \(\epsilon\), grasp-space connectivity is summarized by

$$
B(\epsilon)=\frac{\beta_0(\epsilon)}{N},
\qquad
L(\epsilon)=\frac{|C_{\max}(\epsilon)|}{N},
$$

where \(\beta_0\) is the number of connected components and \(C_{\max}\) is the largest component.

Together, these curves define a multiscale connectivity signature.

---

## E1 — Connectivity Structure and Convergence

E1 tests whether the connectivity signature is a reproducible property of an object's successful-grasp space.

Successful grasps are repeatedly sampled at:

```text
N = 10, 20, 50, 100, 200, 300
```

Connectivity signatures are compared across independent samples. The experiment also compares **within-object** and **between-object** structural distances at \(N=50,100,300\).

`vis.py` provides MDS visualizations of grasp spaces. Connectivity itself is computed in the original normalized SE(3) space; the 2D projection is used only for visualization.

---

## E2 — Connectivity Reconstruction

E2 asks whether the connectivity structure of a known successful-grasp space can be reconstructed using fewer observations.

Three strategies are compared:

- **Random**
- **FPS** — Farthest Point Sampling
- **Connectivity-aware sampling**

A fixed set of successful grasps defines the reference connectivity signature. This reference is used **only for evaluation** and is never provided to the sampling algorithm.

Default configuration:

```text
Reference grasps: 500
Maximum observations: 200
Objects: 200
Random seeds: 5
```

Run from the repository root:

```bash
python -m src.e2.experiment grasps
```

---

## E3 — Exploration → Exploitation

E3 evaluates whether information acquired during exploration improves later grasp discovery.

For each object, two **disjoint hidden-label pools** are constructed:

- **Pool A:** exploration
- **Pool B:** exploitation

During Phase 1, Random, FPS, and connectivity-aware policies query Pool A and observe success/failure labels only after each query.

During Phase 2, the policies are evaluated on untouched Pool B. Pool-B successes are evaluated but do not become new anchors.

The primary metric is Phase-2 AUC. Improvement over Random is:

$$
G_M =
\frac{A_M-A_{\mathrm{Random}}}
{A_{\mathrm{Random}}}
\times 100\%.
$$

Current experiments evaluate pool sizes:

```text
200, 400, 800, 1600
```

Main analysis scripts:

- `analyze_meta.py` — object-level and cross-scale analysis
- `plot_results.py` — result visualization
- `stats_exploration.py` — exploration statistics
- `stats_failure.py` — dataset success/failure availability
- `stats_final.py` — final aggregate statistics

Run:

```bash
python -m src.e3.experiment grasps
```

Dataset availability can be inspected with:

```bash
python src/e3/stats_failure.py grasps
```

---

## E4

E4 is currently under development.

---

## Shared Utilities

`src/utils/` contains shared functionality:

- `geometry.py` — SE(3) geometry and distances
- `connectivity.py` — connectivity signatures and evaluation
- `grasp_io.py` — grasp dataset I/O
- `samplers.py` — Random, FPS, and connectivity-aware sampling

## Data

Experiments expect ACRONYM-style HDF5 files containing:

```text
grasps/transforms
grasps/qualities/flex/object_in_gripper
```

with binary grasp labels:

```text
1 = successful grasp
0 = failed grasp
```

The grasp dataset and generated results are not included in the repository.

## Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Main dependencies:

```bash
pip install numpy pandas h5py scipy matplotlib scikit-learn
```

## Status

| Experiment | Status |
|---|---|
| E1 | Implemented |
| E2 | Implemented |
| E3 | Implemented |
| E4 | In development |
| Ablations | In development |