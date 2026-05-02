"""Apply two improvements to Hierarchical_MARL_System.ipynb:
1. Add fixed-seed resets before each training call (cells 19, 24, 29, 32, 35).
2. Add per-lambda checkpointing in the ablation cell (32).

Idempotent: safe to re-run.
"""
import json
from pathlib import Path

NB = Path("Hierarchical_MARL_System.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

SEED_HELPER = (
    "def _set_seed(seed=SEED):\n"
    "    np.random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "    if torch.cuda.is_available():\n"
    "        torch.cuda.manual_seed_all(seed)\n"
    "print('Seed helper defined.')\n"
)

# 1. Append seed helper to setup cell (cell 1) if missing
setup_src = "".join(nb["cells"][1]["source"])
if "_set_seed" not in setup_src:
    setup_src = setup_src.rstrip() + "\n\n" + SEED_HELPER
    nb["cells"][1]["source"] = setup_src.splitlines(keepends=True)
    print("Patched cell 1 (added _set_seed helper)")

# Helper: prepend a seed call to a cell if not already present
def prepend_seed(cell_idx, marker_comment):
    src = "".join(nb["cells"][cell_idx]["source"])
    if "_set_seed(SEED)" in src:
        print(f"Cell {cell_idx}: seed already present, skipping")
        return
    new_src = f"{marker_comment}\n_set_seed(SEED)\n\n" + src
    nb["cells"][cell_idx]["source"] = new_src.splitlines(keepends=True)
    print(f"Patched cell {cell_idx}: prepended _set_seed")


prepend_seed(19, "# Reproducibility: reset seeds before concurrent training")
prepend_seed(24, "# Reproducibility: reset seeds before staged training")
prepend_seed(29, "# Reproducibility: reset seeds before walk-forward")
prepend_seed(35, "# Reproducibility: reset seeds before random-control experiment")

# 2. Patch ablation cell (32): per-lambda checkpointing + seed reset per (mode,lambda)
ablation_old = "".join(nb["cells"][32]["source"])
if "ablation_checkpoint" not in ablation_old:
    new_ablation = '''# Reproducibility: reset seed before ablation, and per-lambda for fairness across modes
_set_seed(SEED)

import os, pickle
CKPT_PATH = "data/processed/ablation_checkpoint.pkl"
os.makedirs("data/processed", exist_ok=True)

lambdas = [0.0, 0.1, 0.35, 0.5, 1.0]

# Resume support: load any partial results from a previous interrupted run
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH, "rb") as f:
        ablation_results = pickle.load(f)
    done_c = list(ablation_results.get("concurrent", {}).keys())
    done_s = list(ablation_results.get("staged", {}).keys())
    print(f"Resuming ablation. Already done concurrent={done_c} staged={done_s}")
else:
    ablation_results = {"concurrent": {}, "staged": {}}

t_abl = time.time()
for lam in lambdas:
    print(f"\\n==== Lambda = {lam} ====")

    # ---- Concurrent ----
    if lam in ablation_results["concurrent"]:
        print(f"  Concurrent  [cached] {ablation_results['concurrent'][lam].get('sharpe', 0):+.3f}")
    else:
        _set_seed(SEED + int(lam * 1000))  # deterministic per-lambda
        abl_w = {}
        for profile in ['Safe', 'Neutral', 'Risky']:
            pt = risk_pools[profile]
            if len(pt) >= 2:
                abl_w[profile] = WorkerEnv(price_df, lexical_df, pt,
                                           profile=profile.lower(), window_size=30,
                                           lambda_penalty=lam, gamma_penalty=0.01)
        abl_m = ManagerEnv(price_df, lexical_df, risk_pools, window_size=30)
        cn_net, cw_net, _ = train_concurrent(abl_m, abl_w, n_episodes=200, verbose=False)
        cn_eval = evaluate_concurrent(abl_m, abl_w, cn_net, cw_net, max_steps=200)
        ablation_results['concurrent'][lam] = cn_eval
        print(f"  Concurrent  Return={cn_eval['total_return']:+.4f}  Sharpe={cn_eval['sharpe']:+.3f}"
              f"  MaxDD={cn_eval['max_drawdown']:.3f}  V={[f'{a:.2f}' for a in cn_eval['avg_allocation']]}")
        with open(CKPT_PATH, "wb") as f:
            pickle.dump(ablation_results, f)

    # ---- Staged ----
    if lam in ablation_results["staged"]:
        print(f"  Staged      [cached] {ablation_results['staged'][lam].get('sharpe', 0):+.3f}")
    else:
        _set_seed(SEED + int(lam * 1000))
        abl_w_s = {}
        for profile in ['Safe', 'Neutral', 'Risky']:
            pt = risk_pools[profile]
            if len(pt) >= 2:
                abl_w_s[profile] = WorkerEnv(price_df, lexical_df, pt,
                                             profile=profile.lower(), window_size=30,
                                             lambda_penalty=lam, gamma_penalty=0.01)
        abl_m_s = ManagerEnv(price_df, lexical_df, risk_pools, window_size=30)
        sn_net, sw_net, _ = train_staged(abl_m_s, abl_w_s,
                                         phase1_episodes=200, phase2_episodes=160,
                                         phase3_episodes=80, verbose=False)
        sn_eval = evaluate_concurrent(abl_m_s, abl_w_s, sn_net, sw_net, max_steps=200)
        ablation_results['staged'][lam] = sn_eval
        print(f"  Staged      Return={sn_eval['total_return']:+.4f}  Sharpe={sn_eval['sharpe']:+.3f}"
              f"  MaxDD={sn_eval['max_drawdown']:.3f}  V={[f'{a:.2f}' for a in sn_eval['avg_allocation']]}")
        with open(CKPT_PATH, "wb") as f:
            pickle.dump(ablation_results, f)

print(f"\\nAblation complete. ({time.time()-t_abl:.1f}s)")
print(f"Checkpoint saved: {CKPT_PATH}")
'''
    nb["cells"][32]["source"] = new_ablation.splitlines(keepends=True)
    print("Patched cell 32: added checkpointing + per-lambda seed")
else:
    print("Cell 32: checkpointing already present")

# Clear all outputs to keep diff focused on source
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\nWrote {NB} ({NB.stat().st_size} bytes)")
