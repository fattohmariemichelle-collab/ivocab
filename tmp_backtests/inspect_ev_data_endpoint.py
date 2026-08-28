from __future__ import annotations
import re, urllib.request, urllib.parse

BASE='https://evtradelabs.com/data'
HEADERS={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language':'en-US,en;q=0.9',
}

def fetch(url):
    req=urllib.request.Request(url,headers=HEADERS)
    return urllib.request.urlopen(req,timeout=30).read().decode('utf-8','replace')

html=fetch(BASE)
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
        text=fetch(url)
    except Exception as e:
        print('ERR',url,e); continue
    hits=[]
    for p in [r'https?://[^"\'<> ]+', r'[^"\']*\.json\.gz[^"\']*', r'/(?:api|data|datasets|downloads?)/[^"\'<> ]+']:
        hits.extend(re.findall(p,text))
    relevant=sorted(set(h for h in hits if any(k in h.lower() for k in ['json.gz','catalog','download','dataset','nq','storage','supabase','r2'])))
    if relevant:
        print('\nSCRIPT',url,'BYTES',len(text))
        for h in relevant[:300]: print('MATCH',h[:1000])
