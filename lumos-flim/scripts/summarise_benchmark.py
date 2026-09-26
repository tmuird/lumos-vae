"""Merge the split benchmark runs and print the README tables.

    python scripts/summarise_benchmark.py

Realistic rows come from realistic_a/_b/_c.json (the run was split when the
stacked-context VAE was added); the merged set is written to realistic.json.
"""

import json
from pathlib import Path

ROOT = Path("results/benchmark")
ORDER = ["vae", "vae_spatial", "vae_stack", "eb", "tv", "binned", "global", "mle"]
NAMES = {"vae": "VAE", "vae_spatial": "VAE + summed 3x3", "vae_stack": "VAE + stacked 3x3",
         "eb": "empirical Bayes", "tv": "TV-regularised", "binned": "3x3 binned",
         "global": "global lifetimes", "mle": "per-pixel MLE"}


def load(name):
    p = ROOT / name
    return json.load(open(p)) if p.exists() else []


def merge_realistic():
    rows = {}
    for part in ("realistic_a.json", "realistic_b.json", "realistic_c.json"):
        for r in load(part):
            rows[(r["tag"], r["method"])] = r
    merged = sorted(rows.values(), key=lambda r: (r["photons"], ORDER.index(r["method"])))
    json.dump(merged, open(ROOT / "realistic.json", "w"), indent=1)
    return merged


def fmt(v, spec):
    return "—" if v is None or v != v else format(v, spec)


def realistic_table(rows):
    best = {}
    for r in rows:
        best[r["tag"]] = min(best.get(r["tag"], 9e9), r["heldout_nll"])
    out = ["| photons | method | held-out NLL above best (mnats/photon) | τ_amp bias / IQR / r | τ long r | α MAE | τ_amp IQR, edge / interior |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['photons']:.0f} | {NAMES[r['method']]} | {1000 * (r['heldout_nll'] - best[r['tag']]):.1f} | "
            f"{r['all_tau_amp_bias']:+.2f} / {r['all_tau_amp_iqr']:.2f} / {r['all_tau_amp_r']:.2f} | "
            f"{fmt(r['all_tau_long_r'], '.2f') if r['method'] != 'global' else '—'} | {r['all_alpha_mae']:.3f} | "
            f"{r['edge_tau_amp_iqr']:.2f} / {r['interior_tau_amp_iqr']:.2f} |")
    return "\n".join(out)


def embryo_table(rows):
    best = {}
    for r in rows:
        best[r["tag"]] = min(best.get(r["tag"], 9e9), r["heldout_nll"])
    out = ["| photons | method | held-out NLL above best (mnats/photon) | τ_amp vs full-photon MLE: median dev / IQR / r |",
           "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["photons"], ORDER.index(r["method"]))):
        out.append(f"| {r['photons']:.0f} | {NAMES[r['method']]} | {1000 * (r['heldout_nll'] - best[r['tag']]):.1f} | "
                   f"{r['ref_dev_median']:+.3f} / {r['ref_dev_iqr']:.3f} / {r['ref_r']:.2f} |")
    return "\n".join(out)


def tail_table(rows):
    out = ["| photons | method | IRF | held-out NLL (nats/photon) | τ_amp bias / IQR / r | τ short IQR | α MAE | fitted tail |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tail = (f"{r['tail_weight']:.0%}, {r['tail_tau']:.2f} ns" if r.get("tail_weight") is not None else "—")
        out.append(f"| {r['photons']} | {r['method']} | {r['irf']} | {r['heldout_nll']:.4f} | "
                   f"{r['tau_amp_bias']:+.3f} / {r['tau_amp_iqr']:.3f} / {r['tau_amp_r']:.2f} | "
                   f"{r['tau_short_iqr']:.3f} | {r['alpha_mae']:.3f} | {tail} |")
    return "\n".join(out)


def main():
    rows = merge_realistic()
    if rows:
        print("## realistic\n" + realistic_table(rows) + "\n")
    emb = load("embryo.json")
    if emb:
        print("## embryo\n" + embryo_table(emb) + "\n")
    tail = load("irf_tail.json")
    # The Gaussian-IRF rows are the benchmark's own fits on the same split.
    levels = {t["photons"] for t in tail}
    for r in rows:
        n = int(r["tag"].split("_p")[1])
        have = any(t["irf"] == "gaussian" and t["method"].lower() == r["method"] and t["photons"] == n
                   for t in tail)
        if r["method"] in ("vae", "mle") and n in levels and not have:
            tail.append(dict(photons=n, method=r["method"].upper(), irf="gaussian",
                             heldout_nll=r["heldout_nll"],
                             **{k[4:]: v for k, v in r.items() if k.startswith("all_")}))
    tail.sort(key=lambda t: (t["photons"], t["method"], t["irf"]))
    if tail:
        print("## irf tail\n" + tail_table(tail) + "\n")


if __name__ == "__main__":
    main()
