"""Fill the README tables and draw the charts from results/results.jsonl (no hand-copied numbers).

Usage: python scripts/make_report.py   (needs matplotlib; no GPU, no torch)
Tables are written between <!-- table:NAME --> and <!-- /table:NAME --> markers in README.md.
Aggregates across seeds go to results/summary.json, which the README test also accepts as a source.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "results.jsonl"
IMG = ROOT / "docs" / "img"


def load():
    rows = [json.loads(line) for line in RES.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["experiment"]].append(r)
    return by


def latest(rows, key):
    out = {}
    for r in rows:
        out[key(r)] = r
    return list(out.values())


def pct(x):
    return f"{x * 100:.1f}%"


def ci(c):
    return f"{c[0] * 100:.1f} to {c[1] * 100:.1f}"


def put(readme: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- table:{name} -->)(.*?)(<!-- /table:{name} -->)", re.S)
    if not pat.search(readme):
        return readme
    return pat.sub(lambda m: f"{m.group(1)}\n{body.strip()}\n{m.group(3)}", readme)


def main() -> None:
    by = load()
    IMG.mkdir(parents=True, exist_ok=True)
    summary = {}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # ---- E1 linear probe
    e1 = sorted(latest(by["cub_linear_probe"], lambda r: r["model"]), key=lambda r: -r["test_acc"])
    if e1:
        lines = ["| Backbone | Linear-probe top-1 | 95% CI | Feature dim | Extraction, img/s |", "|---|---|---|---|---|"]
        for r in e1:
            lines.append(f"| {r['label']} | {pct(r['test_acc'])} | {ci(r['ci95'])} | {r['feat_dim']} | {r['extract_img_per_s']:.0f} |")
        readme = put(readme, "e1", "\n".join(lines))
        fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(e1) + 1.2))
        labels = [r["label"] for r in e1][::-1]
        acc = [r["test_acc"] * 100 for r in e1][::-1]
        err = [[(r["test_acc"] - r["ci95"][0]) * 100 for r in e1][::-1], [(r["ci95"][1] - r["test_acc"]) * 100 for r in e1][::-1]]
        ax.barh(labels, acc, xerr=err, color="#3b6ea8", capsize=3)
        ax.set_xlim(min(acc) - 5, 100)
        ax.set_xlabel("CUB-200-2011 test top-1, % (frozen features + linear classifier, 95% bootstrap CI)")
        his = [r["ci95"][1] * 100 for r in e1][::-1]
        for y_, (a, h) in enumerate(zip(acc, his)):
            ax.text(h + 0.3, y_, f"{a:.1f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(IMG / "e1_probe.png", dpi=150)
        plt.close(fig)

    # ---- E2 fine-tune + ensemble
    e2 = sorted(latest(by["cub_finetune"], lambda r: (r["model"], r.get("tag", ""), r["seed"])), key=lambda r: -r["test_acc"])
    if e2:
        lines = ["| Model | Recipe | Top-1 | 95% CI | Epochs | Training time | Train img/s | Peak GPU memory |", "|---|---|---|---|---|---|---|---|"]
        for r in e2:
            lines.append(f"| {r['label']} | {r.get('recipe', 'naive')} | {pct(r['test_acc'])} | {ci(r['ci95'])} | {r['epochs']} | {r['train_minutes']:.1f} min | {r['train_img_per_s']:.0f} | {r['peak_gpu_mem_gb']:.1f} GB |")
        ens = by.get("cub_ensemble", [])
        if ens:
            e = ens[-1]
            lines.append(f"| Ensemble: {' + '.join(m.replace(' fine-tuned', '') for m in e['members'])} | equal-weight average | {pct(e['test_acc'])} | {ci(e['ci95'])} | | | | |")
            summary["ensemble_gain_pp"] = round(e["gain"] * 100, 2)
            summary["ensemble_gain_ci_pp"] = [round(e["gain_ci95"][0] * 100, 2), round(e["gain_ci95"][1] * 100, 2)]
        readme = put(readme, "e2", "\n".join(lines))
        if ens:
            e = ens[-1]
            readme = put(readme, "e2gain", f"Ensemble gain over the best single model ({e['best_single']}): **{e['gain'] * 100:+.2f} pp**, paired-bootstrap 95% CI {e['gain_ci95'][0] * 100:+.2f} to {e['gain_ci95'][1] * 100:+.2f} pp, share of resamples where it is not better: {e['p_not_better']:.3f}.")
    cmp_rows = latest(by.get("cub_compare", []), lambda r: (r["model"], r["a"], r["b"]))
    if cmp_rows:
        lines = ["| Backbone | Comparison (same test images) | Difference | Paired 95% CI | Verdict |", "|---|---|---|---|---|"]
        for r in cmp_rows:
            lo, hi = r["diff_ci95"]
            verdict = "real gain" if lo > 0 else "real loss" if hi < 0 else "no measurable difference"
            lines.append(f"| {r['label']} | {r['b']} vs {r['a']} | {r['diff'] * 100:+.2f} pp | {lo * 100:+.2f} to {hi * 100:+.2f} pp | {verdict} |")
        readme = put(readme, "e2cmp", "\n".join(lines))

    # ---- E3 throughput + autotune
    e3 = latest(by["throughput"], lambda r: (r.get("task", "cub"), r["loader"]))
    if e3:
        work = {"cub": "ConvNeXt-T, 224 px, batch 32 (heavy step)", "gtsrb": "ResNet-50, 64 px signs, batch 256 (light step)"}
        lines = ["| Workload | Input pipeline | Training img/s | GPU busy | vs 0 workers |", "|---|---|---|---|---|"]
        for task in ("cub", "gtsrb"):
            rows = [r for r in e3 if r.get("task", "cub") == task]
            base = next((r["img_per_s"] for r in rows if r["mode"] == "files" and r["workers"] == 0), None)
            for r in rows:
                rel = f"{r['img_per_s'] / base:.1f}x" if base else ""
                lines.append(f"| {work[task]} | {r['loader']} | {r['img_per_s']:.0f} | {r['gpu_util_pct']:.0f}% | {rel} |")
            if base and rows:
                summary[f"throughput_{task}"] = {r["loader"]: r["img_per_s"] for r in rows}
        readme = put(readme, "e3", "\n".join(lines))
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.0))
        for ax, task in zip(axes, ("cub", "gtsrb")):
            rows = [r for r in e3 if r.get("task", "cub") == task]
            if not rows:
                ax.axis("off")
                continue
            ax.barh([r["loader"].replace(" DataLoader", "") for r in rows][::-1], [r["img_per_s"] for r in rows][::-1],
                    color=["#c0504d" if r["mode"] == "files" else "#3b6ea8" for r in rows][::-1])
            ax.set_title(work[task], fontsize=9)
            ax.set_xlim(0, max(r["img_per_s"] for r in rows) * 1.55)
            ax.set_xlabel("training images/s")
            for y_, r in enumerate(rows[::-1]):
                ax.text(r["img_per_s"], y_, f" {r['img_per_s']:.0f} (GPU {r['gpu_util_pct']:.0f}%)", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(IMG / "e3_throughput.png", dpi=150)
        plt.close(fig)
    e3b = latest(by["cudnn_autotune"], lambda r: r["label"])
    if e3b:
        lines = ["| ConvNeXt-T forward, batch 64 | ms per batch | img/s |", "|---|---|---|"]
        for r in e3b:
            lines.append(f"| {r['label']} | {r['ms_per_batch']:.0f} | {r['img_per_s']:.0f} |")
        readme = put(readme, "e3b", "\n".join(lines))

    # ---- E4 CPU budget
    e4 = latest(by["cpu_budget"], lambda r: r["model"])
    if e4:
        acc_of = {r["model"]: r["test_acc"] for r in by["cub_linear_probe"]}
        ft = {r["model"]: r["test_acc"] for r in by["cub_finetune"]}
        lines = ["| Model | Params | fp32 weights | CPU latency, 1 image | CPU throughput, batch 32 | Whole test split on CPU | Best CUB top-1 |", "|---|---|---|---|---|---|---|"]
        pts = []
        for r in sorted(e4, key=lambda r: r["latency_ms_b1"]):
            a = ft.get(r["model"], acc_of.get(r["model"]))
            src = "fine-tuned" if r["model"] in ft else "probe"
            lines.append(f"| {r['label']} | {r['params_m']:.1f} M | {r['weights_mb']:.0f} MB | {r['latency_ms_b1']:.0f} ms | {r['batch32_img_per_s']:.0f} img/s | {r['cub_test_seconds']:.0f} s | {pct(a)} ({src}) |" if a else f"| {r['label']} | {r['params_m']:.1f} M | {r['weights_mb']:.0f} MB | {r['latency_ms_b1']:.0f} ms | {r['batch32_img_per_s']:.0f} img/s | {r['cub_test_seconds']:.0f} s | |")
            if a:
                pts.append((r["latency_ms_b1"], a * 100, r["label"]))
        readme = put(readme, "e4", "\n".join(lines))
        if pts:
            fig, ax = plt.subplots(figsize=(7.5, 4))
            for x_, y_, lab in pts:
                ax.scatter(x_, y_, color="#3b6ea8")
                ax.annotate(lab, (x_, y_), textcoords="offset points", xytext=(5, 3), fontsize=8)
            ax.set_xscale("log")
            ax.set_xlabel("CPU latency per image, ms (8 threads, fp32, log scale)")
            ax.set_ylabel("CUB-200 top-1, %")
            fig.tight_layout()
            fig.savefig(IMG / "e4_cpu.png", dpi=150)
            plt.close(fig)

    # ---- E5 GTSRB held-out classes
    e5 = by.get("gtsrb_unseen", [])
    if e5:
        groups = defaultdict(list)
        for r in latest(e5, lambda r: (r["condition"], r["seed"])):
            groups[r["condition"]].append(r)
        order = ["oracle", "real_seen_only", "synth_unseen_only", "synth_all", "synth_all + knn"]
        names = {
            "oracle": "Real images of all 43 classes (upper bound)",
            "real_seen_only": "Real images of the 33 seen classes only",
            "synth_unseen_only": "+ synthetic images of the 10 held-out classes",
            "synth_all": "+ synthetic images of all 24 drawable classes",
            "synth_all + knn": "same network, cosine k-NN head",
        }
        lines = ["| Training data | Runs | Overall | Seen classes | Held-out classes (0 real images), mean ± SD over runs | Held-out 95% bootstrap CI, seed 0 |", "|---|---|---|---|---|---|"]
        s5 = {}
        for c in order:
            g = groups.get(c)
            if not g:
                continue
            mo = statistics.mean(r["overall"] for r in g)
            ms = statistics.mean(r["seen"] for r in g)
            mu = statistics.mean(r["unseen"] for r in g)
            sd = statistics.pstdev(r["unseen"] for r in g) if len(g) > 1 else 0.0
            g0 = min(g, key=lambda r: r["seed"])
            s5[c] = {"overall": round(mo, 4), "seen": round(ms, 4), "unseen": round(mu, 4), "unseen_sd": round(sd, 4), "runs": len(g)}
            spread = f" ± {sd * 100:.1f}" if len(g) > 1 else ""
            lines.append(f"| {names[c]} | {len(g)} | {pct(mo)} | {pct(ms)} | {pct(mu)}{spread} | {ci(g0['unseen_ci95'])} |")
        summary["gtsrb"] = s5
        readme = put(readme, "e5", "\n".join(lines))
        bestc = max((c for c in s5 if c != "oracle"), key=lambda c: s5[c]["unseen"], default=None)
        if bestc:
            per = defaultdict(list)
            for r in groups[bestc]:
                for k, v in r["per_unseen_class"].items():
                    per[k].append(v)
            rows = sorted(((k, statistics.mean(v)) for k, v in per.items()), key=lambda t: -t[1])
            summary["gtsrb_per_heldout_class"] = {"condition": bestc, **{k: round(v, 4) for k, v in rows}}
            readme = put(readme, "e5cls", "\n".join(["| Held-out class | Accuracy (mean over runs) |", "|---|---|"] + [f"| {k} | {pct(v)} |" for k, v in rows]))
            fig, ax = plt.subplots(figsize=(8.5, 3.4))
            ax.barh([k for k, _ in rows][::-1], [v * 100 for _, v in rows][::-1], color="#3b6ea8")
            ax.set_xlabel("Accuracy on real test photos, %, for classes trained only on rendered signs (mean of runs)")
            ax.set_xlim(0, 100)
            fig.tight_layout()
            fig.savefig(IMG / "e5_heldout.png", dpi=150)
            plt.close(fig)

    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("README tables and charts updated")


if __name__ == "__main__":
    main()
