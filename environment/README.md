# Environment

Results in the paper were produced with Python 3.10.4, numpy 1.26.4, scipy 1.15.3,
scikit-learn 1.3.2, pandas 2.3.3 on Windows 10 (see `requirements.txt`).

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r environment/requirements.txt
```

The reproduction script was also run on a newer stack (see `docs/REPRODUCIBILITY.md`
for the measured differences). GP hyperparameter optimization (L-BFGS-B with restarts)
and the MLP (Adam, early stopping) are sensitive to library versions at the level of
a few 1e-3 dB; the pinned versions above are the reference.
