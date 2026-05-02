import json
nb = json.load(open('Hierarchical_MARL_System.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    for out in c.get('outputs', []):
        if out.get('output_type') == 'error':
            print(f"=== ERROR in cell {i} ===")
            print('Source first line:', ''.join(c['source']).splitlines()[0][:120])
            print('ename:', out.get('ename'))
            print('evalue:', out.get('evalue'))
            tb = out.get('traceback', [])
            print('\n'.join(tb[-15:]))
            print()
