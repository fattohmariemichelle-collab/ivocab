from __future__ import annotations
import re, urllib.request, urllib.parse

BASE='https://evtradelabs.com/data'
html=urllib.request.urlopen(BASE,timeout=30).read().decode('utf-8','replace')
print('HTML_BYTES',len(html))
patterns=[r'https?://[^"\'<> ]+',r'/(?:api|data|datasets|downloads?)/[^"\'<> ]+']
for p in patterns:
    for m in sorted(set(re.findall(p,html))):
        if any(k in m.lower() for k in ['json','catalog','download','dataset','api','nq']): print('HTML_MATCH',m[:500])
srcs=re.findall(r'<script[^>]+src=["\']([^"\']+)',html)
print('SCRIPTS',len(srcs))
for src in srcs:
    url=urllib.parse.urljoin(BASE,src)
    try:
        text=urllib.request.urlopen(url,timeout=30).read().decode('utf-8','replace')
    except Exception as e:
        print('ERR',url,e); continue
    hits=[]
    for p in [r'https?://[^"\'<> ]+', r'[^"\']*\.json\.gz[^"\']*', r'/(?:api|data|datasets|downloads?)/[^"\'<> ]+']:
        hits.extend(re.findall(p,text))
    relevant=sorted(set(h for h in hits if any(k in h.lower() for k in ['json.gz','catalog','download','dataset','nq','storage','supabase','r2'])))
    if relevant:
        print('\nSCRIPT',url,'BYTES',len(text))
        for h in relevant[:200]: print('MATCH',h[:1000])
