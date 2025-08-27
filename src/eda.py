from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import Counter
from scipy import stats as scistats
import itertools


def summarize_dataset(ds):
    print("Total records:", len(ds))
    # unique alert_ids
    unique_ids = set(r["alert_id"] for r in ds.records)
    print("Unique alert_ids:", len(unique_ids))

    # presencia por sensor
    both = sum(1 for r in ds.records if r["s2"] and r["s1"])
    only_s2 = sum(1 for r in ds.records if r["s2"] and not r["s1"])
    only_s1 = sum(1 for r in ds.records if r["s1"] and not r["s2"])
    print(f"Records with both S2&S1: {both}")
    print(f"Only S2: {only_s2} | Only S1: {only_s1}")

    # S2 providers
    s2_prov = [r["s2"].provider for r in ds.records if r["s2"]]
    print("S2 providers:", Counter(s2_prov))

    # S1–S2 temporal deltas (days)
    deltas = []
    for r in ds.records:
        if r["s2"] and r["s1"]:
            delta = r["s2"].date - r["s1"].date
            if isinstance(delta, np.timedelta64):
                d = abs(delta / np.timedelta64(1, 'D'))
            elif hasattr(delta, 'days'):
                d = abs(delta.days)
            else:
                d = abs(int(delta))
            deltas.append(d)
    if deltas:
        print(f"|Δ days| S1–S2: median={np.median(deltas):.1f}, P90={np.percentile(deltas,90):.1f}, max={np.max(deltas)}")
    else:
        print("|Δ days| S1–S2: n/a")


def plot_pairing_hist(records):
    vals = []
    for r in records:
        if r["s2"] and r["s1"]:
            delta = r["s2"].date - r["s1"].date
            if isinstance(delta, np.timedelta64):
                d = abs(delta / np.timedelta64(1, 'D'))
            elif hasattr(delta, 'days'):
                d = abs(delta.days)
            else:
                d = abs(int(delta))
            vals.append(d)
    if not vals:
        print("No paired records.")
        return
    plt.figure(figsize=(6,3))
    plt.hist(vals, bins=min(30, max(5, len(set(vals)))))
    plt.title("|Δ days| Between S1 and S2")
    plt.xlabel("days")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_valid_hist(ds, n=200):
    vals = [ ds[i]["valid"].numpy().mean() for i in range(min(n, len(ds))) ]
    if not vals: return
    plt.figure(figsize=(6,3))
    plt.hist(vals, bins=20)
    plt.title("Valid Fraction per Sample (QA/Clouds)")
    plt.xlabel("Fraction")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def quick_panel(sample: Dict, output: Optional[str]=None, prange_rgb=(2,98), prange_sar=(2,98)):
    chans = sample["channels"]
    img = sample["image"].numpy()
    mask = sample["mask"].numpy()
    valid = sample["valid"].numpy()
    qc = sample.get("quality", {})
    cp = qc.get("clean_percent", None)

    dates = sample.get("dates", {})
    idx = sample.get("alert_id", "unknown")

    def get(name): return img[chans.index(name)] if name in chans else None

    rgb = make_rgb_from_sample(sample, prange=prange_rgb, gamma=1.2)

    vv = get("S1_VV")
    vh = get("S1_VH")
    ratio = get("S1_VV_VH_RATIO")
    if vv is not None: vv = normalize_for_display(vv, prange_sar, valid=valid)
    if vh is not None: vh = normalize_for_display(vh, prange_sar, valid=valid)
    if ratio is not None: ratio = normalize_for_display(ratio, (2,98), valid=valid)

    ndvi = get("S2_NDVI")
    if ndvi is not None: ndvi = np.clip(ndvi, -1, 1)

    # compila paneles disponibles
    title_suffix = []
    if "S2" in dates: title_suffix.append(f"S2:{dates['S2'][:10]}")
    if "S1" in dates: title_suffix.append(f"S1:{dates['S1'][:10]}")
    if cp is not None:
        title_suffix.append(f"clean={cp:.1f}%")
    title_suffix = "  |  ".join(title_suffix)

    panels = []
    if rgb is not None: panels.append(("S2 RGB", rgb))
    if ndvi is not None: panels.append(("S2 NDVI", ndvi))
    if vv is not None: panels.append(("S1 VV (disp)", vv))
    if vh is not None: panels.append(("S1 VH (disp)", vh))
    if ratio is not None: panels.append(("S1 VV–VH (disp)", ratio))
    panels.append(("Mask", np.where(mask==255, np.nan, mask)))
    panels.append(("Valid", valid))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]

    for ax, (title, arr) in zip(axes, panels):
        if arr.ndim == 2:
            im = ax.imshow(arr, cmap="viridis" if title.endswith("(disp)") else None)
            if title in ("Mask","Valid"): 
                im.set_cmap("RdBu_r" if title=="Mask" else "gray")
        else:
            ax.imshow(arr)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Alert {idx}   {title_suffix}", fontsize=12)
    plt.tight_layout()
    if output: 
        plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.show()
    

def channel_summary(dataset, n=100):
    import random
    sel = random.sample(range(len(dataset)), min(n, len(dataset)))
    mins, maxs, means = {}, {}, {}
    for i in sel:
        s = dataset[i]
        x = s["image"].numpy()
        chans = s["channels"]
        for ci, name in enumerate(chans):
            arr = x[ci]
            mins[name] = min(mins.get(name, float('inf')), float(arr.min()))
            maxs[name] = max(maxs.get(name, float('-inf')), float(arr.max()))
            means.setdefault(name, []).append(float(arr.mean()))
    print("\nChannel ranges (subset):")
    for name in sorted(means.keys()):
        print(f"  {name:14s} min={mins[name]:.3f} max={maxs[name]:.3f} mean={np.mean(means[name]):.3f}")


def _percentile_stretch(a, p=(2, 98), valid=None, gamma=1.0):
    """Scale to 0-1 using percentiles (per channel)."""
    x = a.astype(np.float32).copy()
    if valid is not None:
        v = x[valid]
        if v.size < 10:  # por si hay poco válido
            v = x[np.isfinite(x)]
    else:
        v = x[np.isfinite(x)]
    if v.size == 0:
        return np.clip(x, 0, 1)
    lo, hi = np.percentile(v, p)
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max() + 1e-6)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    if gamma != 1.0:
        x = np.clip(x, 0, 1) ** (1.0 / gamma)
    return x


def make_rgb_from_sample(sample, prange=(2,98), gamma=1.2):
    """Construct RGB (B04,B03,B02) with percentile stretching over valid pixels."""
    chans = sample["channels"]
    img = sample["image"].numpy()
    valid = sample["valid"].numpy()
    need = ["S2_B04","S2_B03","S2_B02"]
    if not all(n in chans for n in need):
        return None
    r = img[chans.index("S2_B04")]
    g = img[chans.index("S2_B03")]
    b = img[chans.index("S2_B02")]
    r = _percentile_stretch(r, prange, valid=valid, gamma=gamma)
    g = _percentile_stretch(g, prange, valid=valid, gamma=gamma)
    b = _percentile_stretch(b, prange, valid=valid, gamma=gamma)
    rgb = np.stack([r,g,b], axis=-1)
    return np.clip(rgb, 0, 1)


def normalize_for_display(a, prange=(2,98), valid=None):
    """Normalizes any band (e.g. S1 VV/VH in dB) to 0-1 for imshow."""
    return _percentile_stretch(a, prange, valid=valid, gamma=1.0)


def band_stats_by_provider(ds, bands=("S2_B04","S2_B03","S2_B02","S2_B08"), n_samples=300):
    # groups by provider using random samples from the dataset
    idxs = np.random.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    groups = {}  # provider -> {band: [means]}
    for i in idxs:
        s = ds[i]
        prov = s.get("provider", None)
        # if provider not saved in sample, infer from record
        if prov is None:
            r = ds.records[i]
            prov = r["s2"].provider if r["s2"] else "NO_S2"
        x = s["image"].numpy()
        chans = s["channels"]
        for b in bands:
            if b in chans:
                val = x[chans.index(b)].mean()
                groups.setdefault(prov, {}).setdefault(b, []).append(float(val))

    # violin + stats
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(len(bands)+1, 1, hspace=0.5)

    for bi, b in enumerate(bands):
        ax = fig.add_subplot(gs[bi, 0])
        all_data, labels = [], []
        for prov, d in groups.items():
            if b in d and len(d[b]) > 5:
                all_data.append(np.array(d[b]))
                labels.append(f"{prov}\n(n={len(d[b])})")
        if not all_data: 
            ax.set_title(f"{b}: w/o data")
            ax.axis("off")
            continue
        parts = ax.violinplot(all_data, positions=range(len(all_data)), widths=0.7,
                              showmeans=True, showmedians=True)
        for body in parts['bodies']: body.set_alpha(0.6)
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('black')
        ax.boxplot(all_data, positions=range(len(all_data)), widths=0.25)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(f"{b} - distribution by provider")
        ax.grid(True, axis='y', alpha=0.3)
    ax = fig.add_subplot(gs[-1, 0])
    ax.axis('off')

    # simple comparison table
    rows = []
    for prov, d in groups.items():
        for b in bands:
            if b in d and len(d[b])>5:
                arr = np.array(d[b])
                rows.append((prov, b, arr.mean(), arr.std(), arr.min(), arr.max(),
                             scistats.skew(arr), scistats.kurtosis(arr)))
    print("Summary by provider/band (mean +/- std, min–max):")
    for r in rows:
        prov, b, m, s, mn, mx, sk, ku = r
        print(f"  {prov:12s} {b:7s}  μ={m:.3f} σ={s:.3f}  [{mn:.3f},{mx:.3f}]  skew={sk:.2f}  kurt={ku:.2f}")

    # ANOVA by band
    print("\nANOVA / Kruskal by band:")
    for b in bands:
        data = [np.array(groups[p][b]) for p in groups if b in groups[p] and len(groups[p][b])>5]
        if len(data) >= 2:
            F, p = scistats.f_oneway(*data)
            H, p_kw = scistats.kruskal(*data)
            print(f"  {b}: ANOVA p={p:.4g} | Kruskal p={p_kw:.4g}")


def plot_quality(ds, n=300):
    vals_i, vals_u, vals_s2, vals_s1 = [], [], [], []
    for i in range(min(n, len(ds))):
        q = ds[i]["quality"]
        vals_i.append(q["clean_percent"])
        vals_u.append(q["clean_union_percent"])
        if q["s2_clean_percent"] is not None: vals_s2.append(q["s2_clean_percent"])
        if q["s1_clean_percent"] is not None: vals_s1.append(q["s1_clean_percent"])

    plt.figure(figsize=(10,3))
    for k, vals in enumerate([vals_i, vals_u, vals_s2, vals_s1]):
        if not vals: continue
        plt.subplot(1,4,k+1)
        plt.hist(vals, bins=20)
        plt.title(["intersect","union","S2","S1"][k])
        plt.xlabel("%")
        plt.ylabel("count")
    plt.tight_layout()
    plt.show()