import csv
import os
from datetime import datetime
import numpy as np
import config

EXP_HEADERS = [
    "timestamp","trial_no","test_case",
    "ear_score","texture_score","texture_method","challenge_score",
    "final_score","fusion_method","predicted","ground_truth","correct",
]
GROUND_TRUTH = {"real":"REAL","photo":"FAKE","video":"FAKE"}


def load_results():
    if not os.path.exists(config.RESULTS_CSV): return []
    with open(config.RESULTS_CSV,newline="") as f:
        return list(csv.DictReader(f))


def save_result(liveness_result, test_case):
    rows    = load_results()
    gt      = GROUND_TRUTH[test_case]
    correct = int(liveness_result.is_real == (gt=="REAL"))
    row = {
        "timestamp"      : datetime.now().isoformat(timespec="seconds"),
        "trial_no"       : len(rows)+1,
        "test_case"      : test_case,
        "ear_score"      : f"{liveness_result.ear_score:.3f}",
        "texture_score"  : f"{liveness_result.texture_score:.3f}",
        "texture_method" : liveness_result.texture_method,
        "challenge_score": f"{liveness_result.challenge_score:.3f}",
        "final_score"    : f"{liveness_result.final_score:.3f}",
        "fusion_method"  : liveness_result.fusion_method,
        "predicted"      : liveness_result.label(),
        "ground_truth"   : gt,
        "correct"        : correct,
    }
    exists = os.path.exists(config.RESULTS_CSV)
    with open(config.RESULTS_CSV,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXP_HEADERS)
        if not exists: w.writeheader()
        w.writerow(row)
    print(f"[EXP] Trial #{row['trial_no']} — "
          f"predicted={row['predicted']} gt={gt} correct={bool(correct)} "
          f"fusion={row['fusion_method']} texture={row['texture_method']}")


def print_report():
    rows = load_results()
    if not rows: print("No experiment results yet."); return
    cases = ["real","photo","video"]
    print(f"\n── Experiment Report {'─'*32}")
    print(f"{'Case':<12}{'Trials':>6}{'Correct':>8}{'Accuracy':>10}")
    print("─"*40)
    total_t=total_c=0
    for case in cases:
        sub=[r for r in rows if r["test_case"]==case]
        t=len(sub); c=sum(int(r["correct"]) for r in sub)
        if t: print(f"{case:<12}{t:>6}{c:>8}{c/t*100:>9.1f}%")
        else: print(f"{case:<12}{'–':>6}{'–':>8}{'–':>10}")
        total_t+=t; total_c+=c
    print("─"*40)
    if total_t:
        print(f"{'OVERALL':<12}{total_t:>6}{total_c:>8}{total_c/total_t*100:>9.1f}%")
    genuine=[r for r in rows if r["ground_truth"]=="REAL"]
    attacks=[r for r in rows if r["ground_truth"]=="FAKE"]
    fn=sum(1 for r in genuine if r["predicted"]=="FAKE")
    fp=sum(1 for r in attacks if r["predicted"]=="REAL")
    bpcer=fn/len(genuine) if genuine else 0.0
    apcer=fp/len(attacks) if attacks else 0.0
    acer=(apcer+bpcer)/2.0
    print(f"\n  APCER : {apcer:.3f}  (attack acceptance rate)")
    print(f"  BPCER : {bpcer:.3f}  (genuine rejection rate)")
    print(f"  ACER  : {acer:.3f}  (average classification error)")

    # Show which AI methods were used
    methods = set(r.get("fusion_method","?") for r in rows)
    tex_methods = set(r.get("texture_method","?") for r in rows)
    print(f"\n  Fusion methods used : {', '.join(methods)}")
    print(f"  Texture methods used: {', '.join(tex_methods)}\n")


def generate_roc():
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        print("Install: pip install matplotlib scikit-learn"); return
    rows = load_results()
    if not rows: print("No results to plot."); return
    scores = np.array([float(r["final_score"]) for r in rows])
    labels = np.array([1 if r["ground_truth"]=="REAL" else 0 for r in rows])
    fpr,tpr,_ = roc_curve(labels,scores)
    roc_auc   = auc(fpr,tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr,tpr,color="steelblue",lw=2,label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("False Positive Rate (APCER)")
    plt.ylabel("True Positive Rate (1 - BPCER)")
    plt.title("Liveness Detection ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png",dpi=150)
    plt.show()
    print("[SAVED] roc_curve.png")
