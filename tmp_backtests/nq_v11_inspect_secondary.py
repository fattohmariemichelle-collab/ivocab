from pathlib import Path
import json
import pandas as pd

root=Path('data_v11_secondary')
rows=[]
for p in sorted(root.rglob('*')):
    if not p.is_file():
        continue
    item={'path':str(p),'size_bytes':p.stat().st_size}
    if p.suffix.lower() in {'.csv','.txt'}:
        for sep in [',',';','\t']:
            try:
                x=pd.read_csv(p,nrows=5,sep=sep)
                if len(x.columns)>1:
                    item['sep']=repr(sep); item['columns']=list(map(str,x.columns)); item['head']=x.astype(str).to_dict(orient='records')
                    break
            except Exception as e:
                item.setdefault('errors',[]).append(type(e).__name__+': '+str(e)[:200])
    rows.append(item)
Path('v11_secondary_inspection.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
