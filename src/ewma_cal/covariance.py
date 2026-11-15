"""Point-in-time EWMA covariance with missing-pair normalization and PSD repair."""

import numpy as np


def estimate_covariances(snapshots, returns, decay=0.94):
    snap = snapshots
    ret = returns
    models = {}
    for n, (day, s) in enumerate(snap.items()):
        names = s.RIC.tolist()[1:]
        x = ret.loc[:day, names].tail(252).to_numpy()
        present = np.isfinite(x)
        N = present.sum(axis=0)
        iv = s.IV.to_numpy()[1:]
        for decay in [decay]:
            wt = decay ** np.arange(len(x) - 1, -1, -1)
            z = np.nan_to_num(x)
            den = (present.T * wt) @ present.astype(float)
            c = (z.T * wt) @ z / np.maximum(den, 1e-12)
            v = np.diag(c).copy()
            v[N < 20] = iv[N < 20] ** 2 / 252
            v = np.maximum(v, 1e-8)
            r = c / np.sqrt(np.outer(v, v))
            known = N >= 20
            off = r[np.ix_(known, known)]
            avg = np.clip(
                (off.sum() - np.trace(off)) / max(1, len(off) * (len(off) - 1)), -0.02, 0.95
            )
            trust = np.minimum(N[:, None], N[None, :]) / (np.minimum(N[:, None], N[None, :]) + 20)
            r = trust * r + (1 - trust) * avg
            np.fill_diagonal(r, 1)
            c = r * np.sqrt(np.outer(v, v))
            vals, U = np.linalg.eigh((c + c.T) / 2)
            c = (U * np.maximum(vals, 1e-10)) @ U.T
            models.setdefault(f"EWMA_{decay}", {})[day] = dict(Names=names, Entry=c, N=N)
    return models[f"EWMA_{decay}"]
