# Register Design — Fixed Layout Solver

A modular Python solver for the **Fixed Layout Register Design** problem:
assign *n* items to *m* pre-specified candidate sites in a 2D workspace so that
the realized pairwise interactions match a target affinity matrix as closely as
possible, subject to hardware constraints.


---

## Problem definition


| Input | Description |
|---|---|
| **Q** | Symmetric *n × n* target affinity matrix (desired pairwise interactions) |
| **Sites** | *m* candidate locations with known *(x, y)* coordinates (*m ≥ n*) |
| **C₆** | Device constant governing the interaction law *U = C₆ / r⁶* |
| **R_min** | Minimum allowed distance between any pair of items |
| **R_max** | Maximum allowed distance from the workspace center |

The goal is to find a one-to-one assignment *σ* of items to sites that minimizes:

```
F(σ) = Σ_{i<j}  w_ij · ( U_{σ(i),σ(j)} − Q_ij )²
```

where *U_st = C₆ / d(s,t)⁶* is the realized interaction between sites *s* and *t*,
precomputed from the known site coordinates.

---


## Installation

**Requirements:** Python 3.10+ and NumPy.

```bash
pip install -r requirements.txt
```


## Running the demo

```bash
python demo.py
```

The demo creates an  instance on a triangular lattice, solves it
with given number of restarts, and prints a full quality report including per-pair error
breakdown and convergence history.

---

## License

This code is provided as-is for educational and research purposes.
