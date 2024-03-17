import json

i = 'metadata'
o = 'ffmeta'

data = json.loads(open(f'{i}.meta.json', 'r').read())

ffmetadata_content = ';FFMETADATA1\n'
for item in data:
    key = item['key']
    value = item['value'].replace('\n', '\n\\\n')
    ffmetadata_content += f"{key}={value}\n"

open(f'{o}.ffmeta', 'w', encoding='utf-8').write(ffmetadata_content)