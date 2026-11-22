# Connectivity-Aware Grasp Exploration

Experimental framework for studying whether the **connectivity structure of robotic grasp spaces in SE(3)** can improve structural reconstruction, exploration, subsequent grasp discovery, and transfer across objects.

The experiments use ACRONYM grasp candidates and compare connectivity-aware sampling with random and geometry-based baselines.

**Paper:** [Connectivity-Aware Exploration of Robotic Grasp Spaces (arXiv:2609.22983)](https://arxiv.org/abs/2609.22983) · [PDF](https://arxiv.org/pdf/2609.22983)

## Project Structure

```text
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
├── e4_revised/
│   ├── ca_random_connectivity_reference_ablation.py
│   ├── structural_retrieval.py
│   ├── tc_cart.py
│   ├── run_e4_reference_memory_size_parallel.sh
│   └── run_e4_prototype_ablation_50.sh
├── e4/  # earlier experiments and controls
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
```

Legacy, sanity-check, and exploratory scripts are omitted from the main workflow where appropriate.

---

## Method

A grasp is represented as a rigid transformation

$$
T \in SE(3).
$$

Pairwise grasp distance combines normalized translational and rotational distances:

$$
d(T_i,T_j)=\sqrt{\left(\frac{d_p}{s_p}\right)^2+\left(\frac{d_R}{s_R}\right)^2}.
$$

The translational term is the Euclidean distance between grasp positions, while the rotational term is the geodesic distance on $SO(3)$.

At each neighborhood radius $\epsilon$, grasp-space connectivity is summarized by

$$
B(\epsilon)=\frac{\beta_0(\epsilon)}{N},\qquad
L(\epsilon)=\frac{|C_{\max}(\epsilon)|}{N},
$$

where $\beta_0$ is the number of connected components and $C_{\max}$ is the largest connected component.

Together, these curves define a multiscale connectivity signature.

---

## E1 — Reproducibility and Object Specificity

E1 tests whether multiscale connectivity is a reproducible and object-specific property of viable grasp spaces.

Successful grasps are independently sampled at:

```text
N = 10, 20, 50, 100, 200, 300
```

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

## E3 — Hidden-Viability Exploration and Exploitation

E3 removes the known-viability assumption and evaluates whether information acquired during exploration improves subsequent grasp discovery.

For each object, two **disjoint hidden-label pools** are constructed:

- **Pool A:** exploration
- **Pool B:** exploitation

During Phase 1, Random, FPS, and connectivity-aware policies query Pool A. Success/failure labels are revealed only after a candidate is queried.

During Phase 2, the information acquired during exploration guides grasp discovery in the untouched Pool B. Pool-B outcomes are used for evaluation and do not become new exploration anchors.

The primary downstream metric is normalized Phase-2 AUC. Improvement over Random is

$$
G_M=\frac{A_M-A_{\mathrm{Random}}}{A_{\mathrm{Random}}}\times 100\%.
$$

Current experiments evaluate candidate-set sizes:

```text
N = 200, 400, 800, 1600
```

with the corresponding query budgets scaling with candidate-set size.

Main analysis scripts:

- `analyze_meta.py` — object-level and cross-scale analysis
- `plot_results_with_uncertainty.py` — result visualization
- `stats_exploration.py` — exploration statistics
- `stats_failure.py` — dataset success/failure availability
- `stats_final_with_uncertainty.py` — aggregate statistics

Run:

```bash
python -m src.e3.experiment grasps
```

Dataset availability can be inspected with:

```bash
python src/e3/stats_failure.py grasps
```

---

## E4 — Cross-Object Structural Memory (Revised)

E4 evaluates whether learned representations of **reference-object grasp regions** improve sequential grasp discovery on **unseen target objects**. Unlike E1–E3, the revised E4 is a supervised reference-to-target transfer experiment with neural memory encoders. It should not be described simply as the earlier CA + random/retrieved-memory weighting control.

### Data separation and online information

- Reference objects provide supervised training labels and positive-grasp prototypes; validation objects are additional disjoint objects (20% of the reference count in the supplied launch scripts). Evaluation targets are disjoint from both.
- All candidate poses of a target object are available as **unlabeled features**. Its success/failure labels are revealed **only after each grasp attempt**.
- For reference training, leave-one-object-out prototype masking prevents an object's own memory from being used to predict its queries. Validation uses fixed sampled observation histories; training samples partial observation histories excluding supervised query indices.
- Reference prototype sets and trained encoder weights remain fixed during target exploration. Only retrieval rankings and observed-target context change online.

### Reference memory and structural retrieval

`tc_cart.py` constructs reference memory from successful grasp regions. Components carry prototypes, component IDs, weights, and bandwidths. Per-object prototype caps control memory size.

`structural_retrieval.py` computes **eight local kNN graph descriptors** from each object's complete *unlabeled* candidate-pose set. Descriptors are attached to successful reference prototype locations; target candidates are compared to reference prototypes **in descriptor space**, rather than by cross-object pose coordinates. This is not graph-isomorphism matching or a learned topology embedding.

The connectivity encoder uses **dynamic outcome-conditioned structural kNN**: after each observed target success/failure, Gaussian positive/negative evidence reranks reference prototype neighbors. This evidence uses only previously attempted target grasps. Default retrieval parameters in the supplied implementation are `--dynamic-strength 0.75`, `--dynamic-bandwidth 1.0`, and `--dynamic-candidates 64`. These are implementation defaults, not established optimal hyperparameters.

### Compared methods

| Method ID | Description |
| --- | --- |
| `random` | Random target candidate selection |
| `target_ca` | Target-only connectivity-aware exploration, without learned reference transfer |
| `random_geometry_encoder` | Static random-prototype geometric memory encoder |
| `fps_geometry_encoder` | Static FPS-prototype geometric memory encoder |
| `geometry_encoder` | Static geometric memory encoder |
| `dynamic_random_geometry_encoder` | Outcome-conditioned geometric retrieval with random prototypes |
| `dynamic_fps_geometry_encoder` | Outcome-conditioned geometric retrieval with FPS prototypes |
| `dynamic_geometry_encoder` | Outcome-conditioned geometric retrieval with geometric prototypes |
| `connectivity_encoder` | Outcome-conditioned **structural-signature** retrieval and learned memory encoder |

When `--dynamic-geometry` is enabled, the three geometric encoders are trained and evaluated with dynamic geometric retrieval, providing a more closely matched control for online outcome conditioning. The encoder memory feature dimension remains nine.

### Main experiments

**Reference-object scaling.** Vary reference/training objects across `6, 12, 25, 50, 100, 200, 400, 800`, using **16 prototypes per reference object**, five independent seeds (`0`–`4`), 500 evaluation targets per seed, and a budget of 100 attempts per target. The supplied launcher uses a fixed reference pool of 1,600, a split seed of 4,242, 30 training epochs, early-stopping patience 8, and `--dynamic-geometry`. Nominal memory size is `reference_objects × 16`; actual prototype counts may be lower.

**Prototype-count ablation.** Fix 50 reference/training objects and vary the per-object prototype cap over `4, 8, 16, 32, 64`. The supplied launcher likewise uses five seeds, 500 targets per seed, 100 attempts, and dynamic geometric controls.

**Timing audit.** Analyze end-to-end online time per attempt, offline training, and reference-memory construction separately. Timing CSVs are needed for offline costs; legacy runtime columns do **not** measure retrieval-only latency.

### Running E4

Place the supplied revised scripts under `src/e4_revised/` with their canonical filenames (remove upload timestamp suffixes), keeping the Python package imports intact. From the repository root, run **one seed per process**:

```bash
bash src/e4_revised/run_e4_reference_memory_size_parallel.sh 0
bash src/e4_revised/run_e4_prototype_ablation_50.sh 0
```

Repeat for seeds `1`–`4` as needed. The reference-scaling launcher writes under `results/e4_reference_scaling_16/`, and the prototype launcher under `results/e4_prototype_ablation_ref50_dynamic_geometry/`. **Use separate output directories for incompatible experiment versions**; the reference-scaling launcher currently uses the same output directory name as some earlier runs.

To run a single configuration directly:

```bash
python -m src.e4_revised.ca_random_connectivity_reference_ablation \
  --grasps-dir grasps --references 50 --reference-pool 1600 \
  --training-mode reference --targets 500 --seeds 0 --split-seed 4242 \
  --budget 100 --max-grasps 1000 --prototypes-per-object 16 \
  --dynamic-geometry --medoids 8 --neighbors 16 --hidden 64 \
  --epochs 30 --train-queries 128 --train-context 99 \
  --batch-size 256 --val-fraction 0.2 --patience 8 --device mps \
  --out results/e4_reference_scaling_16/seed_0/scale_50.csv
```

### Analysis and outputs

- `monitor_e4_preliminary.py`: periodic provisional summaries of **completed** episodes; partial seed runs are not final results.
- `merge_e4_seed.py`: append a finished seed's trajectories and optional timing data to scale-level files. Avoid merging the same seed twice.
- `plot_e4_reference_scaling_main.py`: reference scaling Success@100 plot, AUC tables, seed-level statistics, and complete-run checks.
- `plot_e4_prototype_ablation.py`: prototype-count heatmaps, target-level violin plots, and paired comparisons.
- `analyze_e4_ablations.py`: aggregate scaling/ablation outcomes, paired comparisons, and timing information.
- `e4_memory_timing_analysis.py`: audit full-episode online timing and separately reported offline training/memory-build costs.

The trajectory CSV contains `seed`, `target`, `method`, `step`, `success`, and `cumulative_success`. Key outcome measures are **Success@100** (successful grasps in 100 attempts) and **AUC** (mean cumulative successes across the 100 steps). The analysis scripts check episode completeness and, where applicable, require complete target sets before computing seed-level summaries. Training occurs once per scale/seed configuration; individual target episodes are not independent model-training repetitions.

**Interpretation:** The revised design tests whether structural-signature retrieval outperforms both target-only CA and outcome-conditioned geometric memory under matched budgets. The supplied code and scripts document the protocol, but do not by themselves establish an empirical improvement. Report conclusions only from completed runs.

---

## Ablation Studies

The acquisition-function ablation evaluates the contribution of the individual connectivity-aware components.

The full CA policy is compared with:

- Bridge only
- Frontier only
- Boundary only
- Geometric novelty only

The main scripts are:

```text
src/ablations/e3_structure_ablation.py
src/ablations/plot_structure.py
```

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

```text
grasps/transforms
grasps/qualities/flex/object_in_gripper
```

with binary grasp labels:

```text
1 = successful grasp
0 = failed grasp
```

The grasp dataset and generated experimental results are not included in the repository.

---

## Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install numpy pandas h5py scipy matplotlib scikit-learn torch
```

---

## Citation

If you use this repository, please cite the associated preprint:

**Connectivity-Aware Exploration of Robotic Grasp Spaces.** arXiv:2609.22983 (2026). [Paper](https://arxiv.org/abs/2609.22983) · [PDF](https://arxiv.org/pdf/2609.22983).
