# Connectivity-Aware Grasp Exploration

Experimental framework for studying whether the **connectivity structure of robotic grasp spaces in SE(3)** can improve structural reconstruction, exploration, subsequent grasp discovery, and transfer across objects.

The experiments use ACRONYM grasp candidates and compare connectivity-aware sampling with random and geometry-based baselines.

## Project Structure

~~~text
src/
├── ablations/
│   ├── e3_structure_ablation.py
│   └── plot_structure.py
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
│   ├── plot_results_with_uncertainty.py
│   ├── stats_exploration.py
│   ├── stats_failure.py
│   └── stats_final_with_uncertainty.py
├── e4/
│   ├── building_library.py
│   ├── memory_control.py
│   ├── plot_results.py
│   ├── plot_results_agg.py
│   ├── transfer_experiment_sanity_independ_objects.py
│   └── sanity/
└── utils/
    ├── connectivity.py
    ├── geometry.py
    ├── grasp_io.py
    └── samplers.py
~~~

Legacy, sanity-check, and exploratory scripts are omitted from the main workflow where appropriate.

---

## Method

A grasp is represented as a rigid transformation

$$
T \in SE(3).
$$

Pairwise grasp distance combines normalized translational and rotational distances:

$$
d(T_i,T_j)
=
\sqrt{
\left(\frac{d_p}{s_p}\right)^2+
\left(\frac{d_R}{s_R}\right)^2
}.
$$

The translational term is the Euclidean distance between grasp positions, while the rotational term is the geodesic distance on \(SO(3)\).

At each neighborhood radius \(\epsilon\), grasp-space connectivity is summarized by

$$
B(\epsilon)=\frac{\beta_0(\epsilon)}{N},
\qquad
L(\epsilon)=\frac{|C_{\max}(\epsilon)|}{N},
$$

where \(\beta_0\) is the number of connected components and \(C_{\max}\) is the largest connected component.

Together, these curves define a multiscale connectivity signature.

---

## E1 — Reproducibility and Object Specificity

E1 tests whether multiscale connectivity is a reproducible and object-specific property of viable grasp spaces.

Successful grasps are independently sampled at:

~~~text
N = 10, 20, 50, 100, 200, 300
~~~

Connectivity signatures are compared across independent samples of the same object. The experiment also compares **within-object** and **between-object** structural distances at multiple sample sizes.

For each object, the normalization of the SE(3) distances is computed once from the complete successful-grasp set and held fixed across sample sizes and replicates.

`vis.py` provides metric multidimensional scaling (MDS) visualizations of representative grasp spaces. Connected components are computed in the original normalized SE(3) space; the two-dimensional projection is used only for visualization.

---

## E2 — Known-Viability Structural Reconstruction

E2 asks whether the connectivity structure of a known viable grasp space can be reconstructed using fewer observations.

Three strategies are compared:

- **Random**
- **FPS** — farthest-point sampling
- **CA** — connectivity-aware sampling

A fixed set of successful grasps defines the reference connectivity signature. The complete reference is used **only for evaluation** and is never provided to the sampling methods.

The connectivity-aware acquisition function combines four criteria:

- **Bridge** — favor candidates connecting currently disconnected components
- **Frontier** — probe immediately beyond observed components
- **Boundary** — extend individual components
- **Novelty** — discourage repeated sampling of the same local region

Default configuration:

~~~text
Reference grasps: 500
Maximum observations: 200
Objects: 200
Random seeds: 5
~~~

Run from the repository root:

~~~bash
python -m src.e2.experiment grasps
~~~

---

## E3 — Hidden-Viability Exploration and Exploitation

E3 removes the known-viability assumption and evaluates whether information acquired during exploration improves subsequent grasp discovery.

For each object, two **disjoint hidden-label pools** are constructed:

- **Pool A:** exploration
- **Pool B:** exploitation

During Phase 1, Random, FPS, and connectivity-aware policies query Pool A. Success/failure labels are revealed only after a candidate is queried.

During Phase 2, the information acquired during exploration guides grasp discovery in the untouched Pool B. Pool-B outcomes are used for evaluation and do not become new exploration anchors.

The primary downstream metric is normalized Phase-2 AUC. Improvement over Random is

$$
G_M =
\frac{A_M-A_{\mathrm{Random}}}
{A_{\mathrm{Random}}}
\times 100\%.
$$

Current experiments evaluate candidate-set sizes:

~~~text
N = 200, 400, 800, 1600
~~~

with the corresponding query budgets scaling with candidate-set size.

Main analysis scripts:

- `analyze_meta.py` — object-level and cross-scale analysis
- `plot_results_with_uncertainty.py` — result visualization
- `stats_exploration.py` — exploration statistics
- `stats_failure.py` — dataset success/failure availability
- `stats_final_with_uncertainty.py` — aggregate statistics

Run:

~~~bash
python -m src.e3.experiment grasps
~~~

Dataset availability can be inspected with:

~~~bash
python src/e3/stats_failure.py grasps
~~~

---

## E4 — Cross-Object Structural Memory

E4 tests whether information acquired from previously explored objects can improve exploration of unseen target objects.

Reference and target object sets are disjoint.

A memory library is constructed from reference objects. For each target object, an initial target-specific warm-up is used to estimate its partial connectivity structure. Reference memories are then retrieved according to structural similarity and incorporated into the target exploration policy.

The experiments study:

- target-only connectivity-aware exploration
- memory-assisted connectivity-aware exploration
- memory-library size
- target-versus-memory weighting
- structural-memory retrieval

The structural-memory control compares:

- **CA** — no transferred memory
- **CA + Random Memory** — memory from a randomly selected reference object
- **CA + Retrieved Memory** — memory selected using structural retrieval

This control tests whether improvements arise specifically from retrieving structurally relevant prior experience rather than merely from introducing an additional memory-derived signal.

Current retrieval-control settings include:

~~~text
Library size: M = 200
Memory weights: alpha = 0.10, 0.25, 0.50
Maximum attempts: 400
~~~

Run the retrieval-control experiment from the repository root:

~~~bash
python -m src.e4.memory_control --attempts 400
~~~

Other E4 scripts include:

- `building_library.py` — construct the reference memory library
- `transfer_experiment_sanity_independ_objects.py` — cross-object transfer experiment
- `plot_results.py` — E4 visualization
- `plot_results_agg.py` — aggregated E4 analysis
- `sanity/` — E4 sanity and validation experiments

---

## Ablation Studies

The acquisition-function ablation evaluates the contribution of the individual connectivity-aware components.

The full CA policy is compared with:

- Bridge only
- Frontier only
- Boundary only
- Geometric novelty only

The main scripts are:

~~~text
src/ablations/e3_structure_ablation.py
src/ablations/plot_structure.py
~~~

Run from the repository root using module execution where applicable.

---

## Shared Utilities

`src/utils/` contains shared functionality:

- `geometry.py` — SE(3) geometry and normalized pairwise distances
- `connectivity.py` — connectivity signatures and structural evaluation
- `grasp_io.py` — grasp dataset I/O
- `samplers.py` — Random, FPS, and connectivity-aware sampling

---

## Data

Experiments expect ACRONYM-style HDF5 files containing:

~~~text
grasps/transforms
grasps/qualities/flex/object_in_gripper
~~~

with binary grasp labels:

~~~text
1 = successful grasp
0 = failed grasp
~~~

The grasp dataset and generated experimental results are not included in the repository.

---

## Environment

Create and activate a virtual environment:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

Install the main dependencies:

~~~bash
pip install numpy pandas h5py scipy matplotlib scikit-learn
~~~

---

## Status

| Experiment | Status |
|---|---|
| E1 | Implemented |
| E2 | Implemented |
| E3 | Implemented |
| E4 | Implemented / additional controls in progress |
| Ablations | Implemented |