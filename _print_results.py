import json
r = json.load(open('data/processed/sp500_notebook_results.json', encoding='utf-8'))
c = r['concurrent_evaluation']; s = r['staged_evaluation']
print(f"Concurrent: ret={c['total_return']:+.4f}  sharpe={c['sharpe']:+.3f}  dd={c['max_drawdown']:.3f}")
print(f"Staged    : ret={s['total_return']:+.4f}  sharpe={s['sharpe']:+.3f}  dd={s['max_drawdown']:.3f}")
print()
print("Lambda ablation Sharpe:")
for lam in r['ablation_full']['concurrent']:
    cn = r['ablation_full']['concurrent'][lam]['sharpe']
    st = r['ablation_full']['staged'][lam]['sharpe']
    print(f"  lam={lam}: concurrent={cn:+.3f}   staged={st:+.3f}")
print()
print("Random control:")
for k, v in r['random_control'].items():
    print(f"  {k:8s}: sharpe={v['sharpe']:+.3f}  ret={v['total_return']:+.4f}  dd={v['max_drawdown']:.3f}")
