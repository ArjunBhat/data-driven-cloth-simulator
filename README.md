# Data-Driven Clothing for Interactive Applications

A reduced-order machine-learning pipeline for cloth: a small MLP maps a skeletal pose (14 joint quaternions, 56 values) to per-vertex garment offsets (2,170 vertices x 3 = 6,510 values), optionally through a fixed PCA output layer. The setup follows the one in the Stanford CS229 paper by Kangrui Xue and Jane Wu.

> **Important: synthetic data.** This repository does **not** use real cloth simulations. Poses and "cloth deformations" come from a synthetic generator inside the notebook. The numbers below are **not** a replication of the paper's results and must not be compared with them. Any conclusion (which PCA size is best, how an MLP compares to linear regression) is a property of this generator, not of real cloth. What the notebook does test is the **pipeline**: PCA output layer, train/val/test discipline, early stopping, reproducibility, evaluation and deployment of a saved model.

## Repository contents
| File | Purpose |
|---|---|
| `data_driven_cloth.ipynb` | End-to-end notebook: synthetic data, PCA, training, ablation, diagnostics, latency, artifact saving, inference check |
| `infer_cloth.py` | Standalone inference (NumPy only): loads `artifacts/`, predicts offsets from a new 56-D pose |
| `artifacts/` | **Created by running the notebook** (section 5b): model weights, PCA basis, preprocessing, metadata |
| `requirements.txt` | Python dependencies |

## Models tested
1. **Linear regression** baseline: 56-D pose to 6,510-D offsets.
2. **Direct MLP** (56 -> 128 -> 256 -> 6,510).
3. **MLP + PCA** (56 -> 128 -> 256 -> k, then a fixed linear PCA decoder to 6,510), k in {64, 128, 256}.

Protocol: 10,000 samples, 187 injected "failed simulation" outliers removed (9,813 kept), split 6,000 / 2,000 / 1,813. PCA and all fitting use the training set; the validation set drives early stopping and model selection; the test set is only reported. Three seeds per MLP configuration.

## Results (this run: data seed 42, model seeds 0/1/2, CPU)
MSE per coordinate in synthetic data units squared. MLP rows are mean +/- std over 3 seeds.

| Model | Output | Val | Test |
|---|---|---|---|
| Predict train mean | 6,510 | 7.796e-05 | 7.807e-05 |
| Linear regression | 6,510 | 1.238e-05 | 1.238e-05 |
| MLP 128-256 | 6,510 direct | 3.975e-05 +/- 2.6e-06 | 3.971e-05 +/- 2.6e-06 |
| MLP 128-256 | PCA 64 | 1.244e-05 +/- 2.1e-09 | 1.244e-05 +/- 2.1e-09 |
| MLP 128-256 | PCA 128 | 1.248e-05 +/- 3.5e-09 | 1.247e-05 +/- 2.5e-09 |
| MLP 128-256 | PCA 256 | 1.253e-05 +/- 4.2e-09 | 1.253e-05 +/- 2.9e-09 |

The irreducible noise floor of the generator is 1.225e-05.

How to read this:
- Everything good sits at the noise floor. The synthetic data is close to linearly predictable, so linear regression is essentially tied with the best MLPs. This generator cannot show a large benefit of a non-linear model.
- The PCA output layer helps the MLP a lot over direct 6,510-D output (about 4.0e-05 vs about 1.25e-05). The direct model was still underfitting at the fixed learning rate and was not tuned, so read this as "under this training budget".
- Fewer components scored marginally better (k=64 < 128 < 256), an artefact of the generator (rank <= 56 signal plus i.i.d. noise). It does not imply 64 is right for real data. The differences are about 1-2% of the error.
- Selected configuration: PCA 64, chosen on validation MSE only. The deployed seed is the best-validation seed within that configuration (chosen in code; see section 5b of the notebook).
- Latency (batch of 1, PyTorch, CPU, this machine): about 0.1 ms per pose. This is **not** compared with a physics solver; none was run here.

See the notebook's final markdown cells for the complete list of limitations.

## How to run

### Local
```bash
git clone https://github.com/satvikkalyanpur/data-driven-cloth-simulator.git
cd data-driven-cloth-simulator
pip install -r requirements.txt
jupyter notebook data_driven_cloth.ipynb      # Run All (about 35 min on 2 CPU cores)
```
Running the notebook trains all models and writes `artifacts/`.

### Google Colab
Clone the repository in a cell (`!git clone ...`, then `%cd data-driven-cloth-simulator`), open `data_driven_cloth.ipynb` and choose **Runtime > Run all**. If you upload only the notebook instead, also upload `infer_cloth.py` for section 8.

## Inference demo (no retraining, no PyTorch)
After `artifacts/` exists:
```bash
python infer_cloth.py --random 3 --out pred.npy         # 3 random unit-quaternion poses
python infer_cloth.py --pose my_pose.npy --out pred.npy # (56,), (N,56), (14,4) or (N,14,4); .csv/.txt also accepted
```
```python
from infer_cloth import ClothPredictor
offsets = ClothPredictor("artifacts").predict(pose)     # (2170, 3) for one pose, (N, 2170, 3) for N poses
```
Input handling: each joint quaternion is re-normalised to unit length (a warning is shown if it was not); wrong shapes, NaN/inf and zero-norm quaternions raise an error. The quaternion component order is arbitrary in the synthetic generator. Outputs are synthetic offsets in the generator's data units, not metres and not real cloth.

## Artifacts (`artifacts/`)
- `model_weights.npz`: MLP linear layers.
- `pca.npz`: first-k PCA components, mean, explained-variance ratio (fit on training targets only).
- `preprocessing.json`: input/output conventions (unit-quaternion normalisation only; no feature scaling).
- `metadata.json`: selected configuration, chosen seed, per-seed validation MSE, reported metrics, library versions, synthetic-data note.
