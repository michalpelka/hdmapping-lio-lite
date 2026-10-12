#!/usr/bin/env python3
"""Plot the trajectory of one or more HDMapping-style result folders (mini_lio_run or lidar_odometry_step_1).

  plot_trajectory.py RESULT_DIR[=label] [RESULT_DIR[=label] ...] -o trajectory.png [--csv world.csv]

World pose of each IMU-rate sample = session_poses.mrp[K] * trajectory_lio_K.csv row.
With two or more inputs, the others are compared with the first one on common timestamps: raw difference
(same world frame: gravity aligned, origin at the first pose) and after a best-fit rigid alignment.
Needs numpy and matplotlib only.
"""
import argparse
import os
import re
import sys

import numpy as np


def read_mrp(path):
    with open(path) as f:
        tok = f.read().split("\n")
    n = int(tok[0])
    poses, i = {}, 1
    for _ in range(n):
        name = tok[i].strip()
        m = np.array([[float(v) for v in tok[i + 1 + r].split()] for r in range(4)])
        k = int(re.search(r"(\d+)\D*$", name).group(1))
        poses[k] = m
        i += 5
    return poses


def load_world_trajectory(result_dir):
    scans = read_mrp(os.path.join(result_dir, "session_poses.mrp"))
    ts, Ts = [], []
    for k in sorted(scans):
        path = os.path.join(result_dir, f"trajectory_lio_{k}.csv")
        if not os.path.exists(path):
            continue
        d = np.loadtxt(path, skiprows=1, ndmin=2)
        if d.size == 0:
            continue
        rel = np.zeros((len(d), 4, 4))
        rel[:, :3, :] = d[:, 1:13].reshape(-1, 3, 4)
        rel[:, 3, 3] = 1.0
        ts.append(d[:, 0] * 1e-9)
        Ts.append(scans[k] @ rel)
    if not ts:
        sys.exit(f"no trajectory in {result_dir}")
    t, T = np.concatenate(ts), np.concatenate(Ts)
    o = np.argsort(t, kind="stable")
    return t[o], T[o]


def rigid_fit(a, b):
    """R, t minimising |R a + t - b|."""
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ S @ U.T
    return R, cb - R @ ca


def path_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result folder, optionally DIR=label")
    ap.add_argument("-o", "--out", default="trajectory.png")
    ap.add_argument("--csv", help="also write the world trajectory of the first input (t x y z)")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for r in a.results:
        d, _, label = r.partition("=")
        t, T = load_world_trajectory(d)
        runs.append((label or os.path.basename(os.path.normpath(d)), t, T))
        p = T[:, :3, 3]
        print(f"{runs[-1][0]}: {len(t)} poses, {t[-1] - t[0]:.1f} s, path {path_length(p):.1f} m, "
              f"end {p[-1].round(2).tolist()} (|end-start| {np.linalg.norm(p[-1] - p[0]):.2f} m)")

    if a.csv:
        _, t, T = runs[0]
        np.savetxt(a.csv, np.column_stack([t, T[:, :3, 3]]), fmt="%.9f %.6f %.6f %.6f", header="t_s x y z", comments="")

    stats = []
    ref_label, rt, rT = runs[0]
    for label, t, T in runs[1:]:
        common, ia, ib = np.intersect1d(np.round(rt * 1e6).astype(np.int64), np.round(t * 1e6).astype(np.int64),
                                        return_indices=True)
        if len(common) < 10:
            print(f"{label}: only {len(common)} common timestamps with {ref_label}, no comparison")
            continue
        pa, pb = rT[ia, :3, 3], T[ib, :3, 3]
        raw = np.linalg.norm(pb - pa, axis=1)
        R, tr = rigid_fit(pb, pa)
        al = np.linalg.norm(pb @ R.T + tr - pa, axis=1)
        dR = np.einsum("nji,njk->nik", rT[ia, :3, :3], T[ib, :3, :3])
        ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        s = (f"{label} vs {ref_label} ({len(common)} poses): raw pos diff RMS {np.sqrt((raw**2).mean()):.3f} m, "
             f"max {raw.max():.3f} m, end {raw[-1]:.3f} m; aligned RMS {np.sqrt((al**2).mean()):.3f} m; "
             f"rot diff RMS {np.sqrt((ang**2).mean()):.2f} deg, max {ang.max():.2f} deg")
        print(s)
        stats.append((label, rt[ia] - rt[0], raw, ang))

    nrow = 2 if stats else 1
    fig = plt.figure(figsize=(14, 6 * nrow))
    ax_xy = fig.add_subplot(nrow, 2, 1)
    ax_z = fig.add_subplot(nrow, 2, 2)
    t0 = min(r[1][0] for r in runs)
    for label, t, T in runs:
        p = T[:, :3, 3]
        ax_xy.plot(p[:, 0], p[:, 1], lw=1.2, label=f"{label} ({path_length(p):.0f} m)")
        ax_z.plot(t - t0, p[:, 2], lw=1.2, label=label)
    ax_xy.plot(*runs[0][2][0, :2, 3], "ko", label="start")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set(xlabel="x [m]", ylabel="y [m]", title="top view (world = gravity aligned, origin at first pose)")
    ax_z.set(xlabel="time [s]", ylabel="z [m]", title="height")
    for ax in (ax_xy, ax_z):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if stats:
        ax_d = fig.add_subplot(nrow, 2, 3)
        ax_r = fig.add_subplot(nrow, 2, 4)
        for label, tt, raw, ang in stats:
            ax_d.plot(tt, raw, lw=1, label=label)
            ax_r.plot(tt, ang, lw=1, label=label)
        ax_d.set(xlabel="time [s]", ylabel="m", title=f"position difference to {ref_label} (no alignment)")
        ax_r.set(xlabel="time [s]", ylabel="deg", title=f"rotation difference to {ref_label}")
        for ax in (ax_d, ax_r):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    if a.title:
        fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
