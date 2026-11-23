import re, os
HERE = os.path.dirname(os.path.abspath(__file__))
tx = open(f'{HERE}/results_template.md').read()
tb = open(f'{HERE}/tables_1119.md').read()
parts = re.split(r'<!--TABLE (\w+)-->\n', tb)
d = {parts[i]: parts[i + 1].strip('\n') for i in range(1, len(parts), 2)}
out = re.sub(r'\{\{(\w+)\}\}', lambda m: d[m.group(1)], tx)
open(f'{HERE}/RESULTS_1119.md', 'w').write(out)
print(len(out))
