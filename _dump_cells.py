import json, sys
nb = json.load(open('Hierarchical_MARL_System.ipynb', encoding='utf-8'))
for i in map(int, sys.argv[1:]):
    print(f"===CELL {i} ({nb['cells'][i]['cell_type']})===")
    print(''.join(nb['cells'][i]['source']))
    print()
