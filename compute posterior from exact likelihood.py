# %%
"""Estimate plausible values for two model parameters from count data.

The script tests a grid of parameter pairs in parallel and scores each pair
by how likely it makes the observed counts. It converts the scores into a
probability distribution over the grid, saves the results, and plots the
most probable region covering about 95% of the total probability.

This portfolio version uses ellipses for selected research settings and
the input data path pending publication of the journal article.
"""

import os, gc, time, contextlib
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import compute_Pc_for_combined_signal as pc

# Settings for the count probability calculation.
# C_MAX is the largest count the calculation can represent.
pc.N_FFT = 2048
rho = 0.988
C_MAX = ...

# Skip detailed diagnostics while scoring the grid.
pc.DO_P1_DIAGNOSTICS = False

# Limit the block size of the probability calculation. Smaller blocks
# reduce memory use per call and can take longer.
pc.MODE_BLOCK = 64

# Set the maximum number of likelihood evaluations to run in parallel.
N_JOBS = 30

# Set the search bounds and the number of grid points for each parameter.
A_DM_MIN, A_DM_MAX = ..., ...
A_BG_MIN, A_BG_MAX = ..., ...
N_DM, N_BG = ..., ...

# Floor probabilities at a small positive value to keep their logs finite.
EPS = 1e-300

# Load integer counts, report their range, and reject counts above C_MAX.
C_obs = np.load(...).astype(np.int64)
print(f"Npix={C_obs.size}, C_min={int(C_obs.min())}, C_max={int(C_obs.max())}")
if int(C_obs.max()) > C_MAX:
    raise ValueError(f"Observed C_max={int(C_obs.max())} exceeds C_MAX={C_MAX}. Increase C_MAX.")

# When supported, limit numerical libraries to one thread. Joblib already
# evaluates several pairs at once, so this avoids extra threads inside
# each calculation.
try:
    from threadpoolctl import threadpool_limits
    _threadpool_ctx = threadpool_limits(limits=1)
except Exception:
    _threadpool_ctx = contextlib.nullcontext()

# Open a discard stream for routine messages from the probability function.
_devnull = open(os.devnull, "w")

# Score one parameter pair by summing the log probabilities of the counts.
def loglike_theta(A_DM, A_BG):
    # Redirect standard output while the probability function runs.
    with contextlib.redirect_stdout(_devnull):
        pC = pc.compute_Pc_tot(A_DM=A_DM, A_BG=A_BG, C_max=C_MAX, rho=rho)

    # Look up one probability per observed count and clip before taking logs.
    probs = np.clip(pC[C_obs], EPS, 1.0)
    ll = float(np.sum(np.log(probs)))

    # Release the temporary arrays once the score has been computed.
    del pC, probs
    gc.collect()

    return ll

# Generate evenly spaced candidates for both parameters. Vary A_DM
# fastest so the scores can be reshaped into rows of A_BG values.
A_DM_grid = np.linspace(A_DM_MIN, A_DM_MAX, N_DM)
A_BG_grid = np.linspace(A_BG_MIN, A_BG_MAX, N_BG)
thetas = [(adm, abg) for abg in A_BG_grid for adm in A_DM_grid]

# Score parameter pairs in batches of at most n_jobs. Keep the scores in
# input order so they line up with the parameter grid.
def parallel_in_chunks(thetas, n_jobs):
    out = []
    for s in range(0, len(thetas), n_jobs):
        chunk = thetas[s:s+n_jobs]
        res = Parallel(
            n_jobs=n_jobs,
            backend="threading",
            batch_size=1,
            pre_dispatch=n_jobs,  # Limit how much work is queued up front.
            verbose=0
        )(
            delayed(loglike_theta)(adm, abg) for adm, abg in chunk
        )
        out.extend(res)
        gc.collect()
    return out

t0 = time.perf_counter()
with _threadpool_ctx:
    logL_list = parallel_in_chunks(thetas, N_JOBS)
t1 = time.perf_counter()

print(f"Wall time: {t1 - t0:.2f} s for {len(thetas)} thetas (threading, chunked)")

# Rebuild the grid with A_BG values as rows and A_DM values as columns.
# Give each pair equal prior weight when converting scores to posterior
# probabilities, which represent how plausible each pair is given the data.
logL = np.array(logL_list, dtype=float).reshape(N_BG, N_DM)

# Subtract the largest score before exponentiating to prevent overflow.
logPost = logL - np.max(logL)
Post = np.exp(logPost)
Post /= Post.sum()  # Scale the probabilities so they sum to one.

# Find the cutoff for the most probable 95% region by ranking grid cells.
# Add probabilities from highest to lowest until the total reaches 95%.
flat = Post.ravel()
order = np.argsort(flat)[::-1]
csum = np.cumsum(flat[order])
k = np.searchsorted(csum, 0.95)
thr = flat[order[min(k, len(order)-1)]]
print(f"95% HPD threshold: {thr:.6e}")

# Plot the boundary of the 95% region and save the figure as a PDF.
ADM, ABG = np.meshgrid(A_DM_grid, A_BG_grid)
plt.figure()
plt.contour(ADM, ABG, Post, levels=[thr])
plt.xlabel(r"$A_{\rm DM}$")
plt.ylabel(r"$A_{\rm BG}$")
plt.title("95% HPD contour (exact likelihood, joblib threading)")
plt.tight_layout()
plt.show()
plt.savefig("hpd_contour.pdf", dpi=300, bbox_inches="tight")
# Save the parameter grid, scores, probabilities, and run settings together.
np.savez(
    "posterior_grid_threading.npz",
    A_DM_grid=A_DM_grid,
    A_BG_grid=A_BG_grid,
    logL=logL,
    posterior=Post,
    hpd_threshold=thr,
    N_FFT=pc.N_FFT,
    RHO=rho,
    C_MAX=C_MAX,
    MODE_BLOCK=pc.MODE_BLOCK,
    N_JOBS=N_JOBS,
)
print("Saved: posterior_grid_threading.npz")


# %%
import importlib, inspect
import compute_Pc_for_combined_signal as pc

# Reload the probability module and print the arguments its function accepts.
pc = importlib.reload(pc)
print(inspect.signature(pc.compute_Pc_tot))


