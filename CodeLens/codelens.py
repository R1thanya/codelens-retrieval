#!/usr/bin/env python3
"""Local CPU-first code block retrieval and Git-version index."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sqlite3, subprocess, sys, time
import threading
from collections import Counter, defaultdict
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

EXT = {'.py':'Python','.js':'JavaScript','.jsx':'JavaScript','.ts':'TypeScript','.tsx':'TypeScript','.java':'Java','.kt':'Kotlin','.go':'Go','.rs':'Rust','.c':'C','.h':'C/C++','.cpp':'C++','.cc':'C++','.cs':'C#','.php':'PHP','.rb':'Ruby','.swift':'Swift','.scala':'Scala','.sh':'Shell','.sql':'SQL'}
SKIP={'.git','.venv','venv','node_modules','vendor','dist','build','target','coverage','__pycache__','.next'}
STOP=set('a an the is are was were be been being to of for in on at by with from into before after and or but how what where when why which who does do did can could should would this that it its method class'.split())
TOKEN=re.compile(r'[A-Za-z_$][\w$]*|\d+')
IDENT=re.compile(r'(?<!^)(?=[A-Z])|[_$./\\:-]+')
EMBEDDING_MODEL=os.environ.get('CODELENS_EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2')
QUERY_PREFIX=os.environ.get('CODELENS_QUERY_PREFIX','')
CODE_PREFIX=os.environ.get('CODELENS_CODE_PREFIX','')
FUSION_CONFIG_FILE=Path(os.environ.get('CODELENS_FUSION_CONFIG',Path(__file__).resolve().parent/'codelens_fusion.json'))
_CACHE_ROOT=Path(__file__).resolve().parent/'.cache'/'huggingface'
os.environ.setdefault('HF_HOME',str(_CACHE_ROOT))
os.environ.setdefault('HF_HUB_CACHE',str(_CACHE_ROOT/'hub'))
os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME',str(_CACHE_ROOT/'hub'))
_encoder=None
ENCODER_TIMINGS={'model_load_seconds':0.0,'document_encode_seconds':0.0,'query_encode_seconds':0.0,'documents_encoded':0,'queries_encoded':0}
_SEARCH_CACHE={}
_SEARCH_CACHE_FILE_SIGNATURES={}
_SEARCH_CACHE_LOCK=threading.RLock()

def set_embedding_model(model):
    """Select an encoder before a benchmark run and clear its in-process state."""
    global EMBEDDING_MODEL,_encoder
    EMBEDDING_MODEL=model; _encoder=None
    for key in ENCODER_TIMINGS: ENCODER_TIMINGS[key]=0

def prepare_query(text): return QUERY_PREFIX+str(text)
def prepare_code(text): return CODE_PREFIX+str(text)
def embedding_fingerprint():
    payload='\0'.join((EMBEDDING_MODEL,QUERY_PREFIX,CODE_PREFIX))
    return EMBEDDING_MODEL+'#'+hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

def load_fusion_config(path=None):
    config_path=Path(path) if path else FUSION_CONFIG_FILE
    default={'strategy':'rrf','rrf_k':60,'bm25_weight':0.5,'semantic_weight':0.5,'semantic_cutoff':None}
    if config_path.exists():
        loaded=json.loads(config_path.read_text(encoding='utf-8'))
        default.update(loaded)
    return default

def _ensure_encoder():
    global _encoder
    if _encoder is None:
        from encoder_adapters import load_encoder
        started=time.perf_counter()
        _encoder=load_encoder(EMBEDDING_MODEL,os.environ.get('HF_HUB_CACHE'))
        ENCODER_TIMINGS['model_load_seconds']+=time.perf_counter()-started
    return _encoder

def encoder_info():
    encoder=_ensure_encoder()
    return {'model':EMBEDDING_MODEL,'dimension':encoder.dimension,'parameters':encoder.parameters,'max_tokens':encoder.max_tokens}

def tokens(s):
    out=[]
    for x in TOKEN.findall(s):
        out.extend(y.lower() for y in IDENT.split(x) if y)
        out.append(x.lower())
    return [x for x in out if x not in STOP and len(x)>1]

def db_open(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path); db.row_factory=sqlite3.Row
    db.executescript('''CREATE TABLE IF NOT EXISTS blocks(id INTEGER PRIMARY KEY, revision TEXT, path TEXT, language TEXT, symbol TEXT, kind TEXT, start_line INTEGER, end_line INTEGER, body TEXT, terms TEXT, embedding BLOB, embedding_model TEXT);
    CREATE TABLE IF NOT EXISTS embedding_cache(cache_key TEXT PRIMARY KEY, embedding BLOB);
    CREATE INDEX IF NOT EXISTS blocks_revision ON blocks(revision);''')
    columns={row['name'] for row in db.execute('PRAGMA table_info(blocks)')}
    if 'embedding' not in columns: db.execute('ALTER TABLE blocks ADD COLUMN embedding BLOB')
    if 'embedding_model' not in columns: db.execute('ALTER TABLE blocks ADD COLUMN embedding_model TEXT')
    return db

def encode_texts(texts, batch_size=32, role='document'):
    encoder=_ensure_encoder()
    started=time.perf_counter()
    result=encoder.encode(texts,batch_size=batch_size).astype('float32')
    key='query_encode_seconds' if role=='query' else 'document_encode_seconds'
    count='queries_encoded' if role=='query' else 'documents_encoded'
    ENCODER_TIMINGS[key]+=time.perf_counter()-started; ENCODER_TIMINGS[count]+=len(texts)
    return result

def split_blocks(text, language):
    lines=text.splitlines(); spans=[]
    if language=='Python':
        try:
            import ast
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and getattr(node,'end_lineno',None):
                    first=min([node.lineno,*[d.lineno for d in getattr(node,'decorator_list',[])]])
                    spans.append((first-1,node.end_lineno,node.name,'class' if isinstance(node,ast.ClassDef) else 'function'))
        except SyntaxError:
            spans=[]
        if spans:
            spans=sorted(set(spans))
    starts=[]
    pats=[r'^\s*(?:async\s+)?def\s+([\w]+)',r'^\s*class\s+([\w]+)',r'^\s*(?:export\s+)?(?:async\s+)?function\s+([\w$]+)',r'^\s*(?:export\s+)?class\s+([\w$]+)',r'^\s*(?:public|private|protected|static|async|func|fn|function|def|fun)\b.*?\b([\w$]+)\s*\([^;]*\)\s*(?:\{|:|$)']
    if spans:
        blocks=[(a,b,sym,kind) for a,b,sym,kind in spans]
    else:
        for i,line in enumerate(lines):
            for pattern in pats:
                match=re.match(pattern,line)
                if match:
                    sym=next((g for g in match.groups() if g),None)
                    starts.append((i,sym or 'block','class' if re.search(r'\bclass\b',line) else 'function'))
                    break
        starts=sorted(set(starts))
        if not starts:
            blocks=[(i,min(len(lines),i+60),f'lines_{i+1}','block') for i in range(0,len(lines),48)]
        else:
            blocks=[]
            for n,(a,sym,kind) in enumerate(starts):
                b=starts[n+1][0] if n+1<len(starts) else len(lines)
                blocks.append((a,b,sym,kind))
    result=[]
    for a,b,sym,kind in blocks:
        a=max(0,a); b=min(len(lines),max(a+1,b))
        if b-a>120:
            stride=96
            for part,start in enumerate(range(a,b,stride),1):
                end=min(b,start+120); body='\n'.join(lines[start:end]).strip()
                if len(body)>=20: result.append((f'{sym} [part {part}]',kind,start+1,end,body))
        else:
            body='\n'.join(lines[a:b]).strip()
            if len(body)>=20: result.append((sym,kind,a+1,b,body))
    return result

def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL,text=True).strip()

def index_revision(repo, rev, dbpath):
    repo=Path(repo).resolve()
    try:
        commit=git(repo,'rev-parse',rev); is_git=True
    except (subprocess.CalledProcessError,FileNotFoundError):
        # Also allow indexing a plain source folder, which is useful for a quick local demo.
        is_git=False; commit='workspace' if rev in ('HEAD','workspace') else rev
    db=db_open(dbpath); rows=[]; incremental=False; changed_paths=set(); files_read=0
    if is_git:
        try: parent=git(repo,'rev-parse',f'{commit}^')
        except subprocess.CalledProcessError: parent=None
        parent_present=parent and db.execute('SELECT 1 FROM blocks WHERE revision=? LIMIT 1',(parent,)).fetchone()
        if parent_present:
            incremental=True
            changed_data=subprocess.check_output(['git','-C',str(repo),'diff','--no-renames','--name-only','-z',parent,commit],stderr=subprocess.DEVNULL)
            changed_paths={p for p in changed_data.decode('utf-8',errors='replace').split('\0') if p}
            for old in db.execute('SELECT path,language,symbol,kind,start_line,end_line,body FROM blocks WHERE revision=?',(parent,)):
                if old['path'] in changed_paths: continue
                body=old['body']; rel=old['path']; lang=old['language']; symbol=old['symbol']; kind=old['kind']
                text=f'{rel} {symbol} {kind} {lang} {body}'
                rows.append((commit,rel,lang,symbol,kind,old['start_line'],old['end_line'],body,'',text))
            listing=sorted(changed_paths)
        else: listing=git(repo,'ls-tree','-r','--name-only',commit).splitlines()
    else:
        listing=[]
        for root,dirs,files in os.walk(repo):
            dirs[:]=[d for d in dirs if d not in SKIP and not d.startswith('.')]
            for name in files:
                f=Path(root)/name
                if f.suffix.lower() in EXT and f.stat().st_size<=1_000_000:
                    listing.append(f.relative_to(repo).as_posix())
    for rel in listing:
        p=Path(rel)
        if p.suffix.lower() not in EXT or any(part in SKIP for part in p.parts): continue
        try:
            data=(subprocess.check_output(['git','-C',str(repo),'show',f'{commit}:{rel}'],stderr=subprocess.DEVNULL) if is_git else (repo/rel).read_bytes())
            if len(data)>1_000_000: continue
            text=data.decode('utf-8',errors='replace')
        except Exception: continue
        files_read+=1
        for sym,kind,start,end,body in split_blocks(text,EXT[p.suffix.lower()]):
            terms=tokens(f'{rel} {sym} {kind} {EXT[p.suffix.lower()]} {body}')
            rows.append((commit,rel,EXT[p.suffix.lower()],sym,kind,start,end,body,json.dumps(terms),f'{rel} {sym} {kind} {EXT[p.suffix.lower()]} {body}'))
    texts=[row[-1] for row in rows]
    keys=[hashlib.sha256((embedding_fingerprint()+'\0'+prepare_code(value)).encode('utf-8')).hexdigest() for value in texts]
    cached={}
    for key in set(keys):
        found=db.execute('SELECT embedding FROM embedding_cache WHERE cache_key=?',(key,)).fetchone()
        if found: cached[key]=found['embedding']
    missing=list(dict.fromkeys(key for key in keys if key not in cached))
    if missing:
        lookup={key:value for key,value in zip(keys,texts)}
        encoded=encode_texts([prepare_code(lookup[key]) for key in missing],role='document')
        cached.update({key:vector.tobytes() for key,vector in zip(missing,encoded)})
        with db: db.executemany('INSERT OR IGNORE INTO embedding_cache(cache_key,embedding) VALUES(?,?)',[(key,cached[key]) for key in missing])
    rows=[(*row[:-2],json.dumps(tokens(row[-1])),cached[key],embedding_fingerprint()) for row,key in zip(rows,keys)]
    with db:
        db.execute('DELETE FROM blocks WHERE revision=?',(commit,))
        db.executemany('INSERT INTO blocks(revision,path,language,symbol,kind,start_line,end_line,body,terms,embedding,embedding_model) VALUES(?,?,?,?,?,?,?,?,?,?,?)',rows)
    db.close(); return commit,len(rows),{'incremental':incremental,'changed_paths':len(changed_paths),
        'files_read':files_read,'blocks_indexed':len(rows),'embedding_cache_hits':len(keys)-len(missing),
        'embedding_cache_misses':len(missing)}

def intent_terms(q):
    return list(dict.fromkeys(tokens(q)))

class BM25Index:
    def __init__(self,docs):
        self.n=len(docs); self.df=Counter(); self.postings=defaultdict(list); lengths=[]
        for i,d in enumerate(docs):
            terms=json.loads(d['terms']) if isinstance(d['terms'],str) else d['terms']
            counts=Counter(terms); length=len(terms); lengths.append(length)
            self.df.update(counts.keys())
            for term,freq in counts.items(): self.postings[term].append((i,freq))
        self.avg=sum(lengths)/max(1,self.n) or 1; self.lengths=lengths

    def scores(self,query):
        scores=[0.0]*self.n
        # Repeated query terms add weight, capped to prevent keyword spam.
        for term,query_freq in Counter(tokens(query)).items():
            items=self.postings.get(term,())
            if not items: continue
            idf=math.log(1+(self.n-self.df[term]+.5)/(self.df[term]+.5))
            query_weight=1.0+math.log(min(query_freq,4))
            for i,freq in items:
                length=self.lengths[i]
                scores[i]+=query_weight*idf*(freq*2.2)/(freq+1.2*(.25+.75*length/self.avg))
        return scores

def bm25_scores(query,docs):
    return BM25Index(docs).scores(query)

def reciprocal_rank_fusion(bm25, semantic, rrf_k=60, bm25_weight=0.5, semantic_weight=0.5):
    import numpy as np
    b=np.asarray(bm25,dtype=np.float32); s=np.asarray(semantic,dtype=np.float32); n=len(b)
    lexical_order=np.argsort(-b,kind='stable'); semantic_order=np.argsort(-s,kind='stable')
    lexical_rank=np.zeros(n,dtype=np.int32); semantic_rank=np.empty(n,dtype=np.int32)
    lexical_rank[lexical_order]=np.arange(1,n+1); lexical_rank[b<=0]=0
    semantic_rank[semantic_order]=np.arange(1,n+1)
    denom=(bm25_weight+semantic_weight)/(rrf_k+1)
    return ((bm25_weight*np.where(lexical_rank>0,1/(rrf_k+lexical_rank),0)+semantic_weight/(rrf_k+semantic_rank))/max(denom,1e-12)).tolist()

def fuse_scores(bm25,semantic,config=None):
    import numpy as np
    config=config or load_fusion_config()
    b=np.asarray(bm25,dtype=np.float32); s=np.asarray(semantic,dtype=np.float32)
    if config.get('strategy','rrf')=='minmax':
        def scale(row):
            lo=float(row.min()); hi=float(row.max())
            return (row-lo)/(hi-lo) if hi>lo else np.zeros_like(row)
        result=float(config.get('bm25_weight',.5))*scale(b)+float(config.get('semantic_weight',.5))*scale(s)
    else:
        result=np.asarray(reciprocal_rank_fusion(b,s,rrf_k=int(config.get('rrf_k',60)),
            bm25_weight=float(config.get('bm25_weight',.5)),semantic_weight=float(config.get('semantic_weight',.5))),dtype=np.float32)
    cutoff=config.get('semantic_cutoff')
    if cutoff is not None:
        result[(b<=0)&(s<float(cutoff))]=0.0
    return result.tolist()

def _search_snapshot(query_revision,dbpath,all_versions):
    import numpy as np
    path=str(Path(dbpath).resolve()); signature=None
    try:
        st=os.stat(path); signature=(st.st_mtime_ns,st.st_size)
    except FileNotFoundError: return [],None,None
    scope='*' if all_versions else (query_revision or '')
    key=(path,scope,embedding_fingerprint())
    with _SEARCH_CACHE_LOCK:
        previous_signature=_SEARCH_CACHE_FILE_SIGNATURES.get(path)
        if previous_signature is not None and previous_signature!=signature:
            for old_key in [k for k in _SEARCH_CACHE if k[0]==path]:
                del _SEARCH_CACHE[old_key]
        _SEARCH_CACHE_FILE_SIGNATURES[path]=signature
        cached=_SEARCH_CACHE.get(key)
        if cached and cached[0]==signature: return cached[1],cached[2],cached[3]
        db=db_open(path)
        sql='SELECT * FROM blocks'; params=[]
        if query_revision and not all_versions:
            if len(query_revision)<40:
                got=db.execute('SELECT DISTINCT revision FROM blocks WHERE revision LIKE ?',(query_revision+'%',)).fetchall()
                query_revision=got[0]['revision'] if len(got)==1 else query_revision
                scope=query_revision; key=(path,scope,embedding_fingerprint())
                cached=_SEARCH_CACHE.get(key)
                if cached and cached[0]==signature:
                    db.close(); return cached[1],cached[2],cached[3]
            sql+=' WHERE revision=?'; params=[query_revision]
        docs=[dict(x) for x in db.execute(sql,params)]
        if not docs:
            db.close(); return [],None,None
        vectors=[]
        for d in docs:
            raw=d['embedding']
            vectors.append(np.frombuffer(raw,dtype=np.float32) if raw and d.get('embedding_model')==embedding_fingerprint() else None)
        missing=[i for i,v in enumerate(vectors) if v is None]
        if missing:
            texts=[prepare_code(f"{docs[i]['path']} {docs[i]['symbol']} {docs[i]['kind']} {docs[i]['language']} {docs[i]['body']}") for i in missing]
            encoded=encode_texts(texts,role='document')
            for i,v in zip(missing,encoded):
                docs[i]['embedding']=v.tobytes(); docs[i]['embedding_model']=embedding_fingerprint()
            db.executemany('UPDATE blocks SET embedding=?,embedding_model=? WHERE id=?',[(docs[i]['embedding'],embedding_fingerprint(),docs[i]['id']) for i in missing])
            db.commit()
            for i in missing: vectors[i]=np.frombuffer(docs[i]['embedding'],dtype=np.float32)
        db.close()
        matrix=np.stack(vectors).astype('float32',copy=False)
        lexical=BM25Index(docs)
        try:
            st=os.stat(path); signature=(st.st_mtime_ns,st.st_size)
        except FileNotFoundError: signature=None
        _SEARCH_CACHE_FILE_SIGNATURES[path]=signature
        _SEARCH_CACHE[key]=(signature,docs,lexical,matrix)
        return docs,lexical,matrix

def search(query, revision=None, top_k=10, dbpath='.codelens/index.sqlite', all_versions=False):
    import numpy as np
    docs,bm25,vectors=_search_snapshot(revision,dbpath,all_versions)
    if not docs:return []
    lexical=bm25.scores(query)
    qvec=encode_texts([prepare_query(query)],role='query')[0]
    semantic=vectors@qvec
    fused=fuse_scores(lexical,semantic)
    scored=sorted(zip(fused,docs),key=lambda x:(-x[0],x[1]['path'],x[1]['start_line'],x[1]['revision']))
    groups={}; ordered=[]
    for score,doc in scored:
        body=doc['body']
        if all_versions:
            normalized='\n'.join(line.rstrip() for line in body.strip().splitlines())
            lineage=(doc['path'],doc['symbol'],doc['kind'],hashlib.sha256(normalized.encode('utf-8')).hexdigest())
        else:
            lineage=(doc['revision'],doc['path'],doc['symbol'],doc['start_line'],doc['end_line'])
        if lineage in groups:
            group=groups[lineage]
            if not any(v['revision']==doc['revision'] for v in group['versions']):
                group['versions'].append({'revision':doc['revision'],'score':round(score,5)})
            continue
        item={k:v for k,v in doc.items() if k not in {'embedding','embedding_model','terms'}}
        item['score']=round(score,5); item['snippet']='\n'.join(body.splitlines()[:22])
        item['versions']=[{'revision':doc['revision'],'score':round(score,5)}]
        groups[lineage]=item; ordered.append(item)
    return ordered[:max(1,int(top_k))]

PAGE='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CodeLens — Code retrieval</title><style>
:root{--bg:#f3f7ff;--ink:#14233d;--muted:#667895;--line:#dce7f8;--blue:#4169e1;--panel:#ffffffed}*{box-sizing:border-box}body{margin:0;background-color:var(--bg);background-image:linear-gradient(#4d7ad20d 1px,transparent 1px),linear-gradient(90deg,#4d7ad20d 1px,transparent 1px),radial-gradient(ellipse at 50% 0%,#dceaff 0%,transparent 58%);background-size:36px 36px,36px 36px,auto;background-attachment:fixed;color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.shell{max-width:1100px;margin:auto;padding:40px 28px}.brand{display:flex;align-items:center;gap:12px;font-weight:750;font-size:19px;letter-spacing:-.3px}.mark{width:36px;height:36px;border-radius:12px;background:linear-gradient(145deg,#3264df,#7699ff);color:#fff;display:grid;place-items:center;box-shadow:0 5px 15px #4169e14d}header{display:flex;justify-content:space-between;align-items:center}.badge{border:1px solid #cfdef4;background:#fafdffdc;border-radius:99px;padding:7px 12px;font-size:12px;color:#536987}.hero{margin:56px auto 30px;max-width:780px;text-align:center}.eyebrow{color:#4169e1;text-transform:uppercase;letter-spacing:1.6px;font-size:11px;font-weight:750}.hero h1{font-size:42px;line-height:1.12;letter-spacing:-1.8px;margin:12px 0}.hero p{margin:0;color:var(--muted)}.searchbox{max-width:820px;margin:28px auto 14px;background:#ffffffed;border:1px solid #d5e2f5;border-radius:16px;padding:10px;display:flex;gap:8px;box-shadow:0 12px 34px #244b8517}.searchbox input{border:0;outline:0;flex:1;padding:11px 12px;font:inherit;min-width:120px;background:transparent;color:var(--ink)}.searchbox button{border:0;border-radius:11px;background:linear-gradient(135deg,#3868df,#5c7ef0);color:white;padding:0 20px;font-weight:650;cursor:pointer;box-shadow:0 4px 10px #4169e133}.searchbox button:hover{filter:brightness(1.06);transform:translateY(-1px)}.filters{display:flex;gap:8px;justify-content:center}.filters select{border:1px solid #cfdef4;background:#ffffffed;border-radius:8px;padding:8px 11px;color:#536987}.content{max-width:820px;margin:28px auto}.resulthead{display:flex;align-items:center;justify-content:space-between;color:var(--muted);font-size:13px;margin-bottom:12px}.card{background:var(--panel);backdrop-filter:blur(10px);border:1px solid #d6e2f4;border-radius:16px;padding:18px 20px;margin:13px 0;box-shadow:0 7px 22px #284a7710;transition:transform .16s,box-shadow .16s}.card:hover{transform:translateY(-2px);box-shadow:0 12px 28px #284a771b}.meta{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted)}.lang{color:#4263b4;background:#edf3ff;padding:3px 8px;border-radius:6px}.score{margin-left:auto;min-width:120px;text-align:center;font-size:16px;line-height:1.2;font-weight:800;padding:8px 12px;border-radius:10px;letter-spacing:-.2px}.score small{display:block;font-size:10px;font-weight:650;letter-spacing:.4px;text-transform:uppercase;margin-top:3px;opacity:.82}.score-zero{color:#bd2638;background:#fff0f1;border:1px solid #ffd2d7}.score-low{color:#a35b09;background:#fff7e8;border:1px solid #ffe2ad}.score-good{color:#13744c;background:#e9f8f0;border:1px solid #bdebd1}.history{margin-top:10px;color:#536987;font-size:12px}.history summary{cursor:pointer;font-weight:650}.history ul{margin:6px 0;padding-left:22px}.path{font-weight:750;margin:11px 0 2px;letter-spacing:-.15px}.symbol{color:#4169e1;font:12px ui-monospace,Consolas,monospace}pre{margin:13px 0 0;padding:14px;background:#f4f7fc;border:1px solid #e5ecf7;border-radius:10px;overflow:auto;color:#283a56;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;max-height:310px;white-space:pre-wrap}.empty{text-align:center;padding:48px 16px;color:var(--muted);background:#ffffffc9;border:1px solid #dce7f8;border-radius:16px}.hint{font-size:12px;color:#7285a2;text-align:center;margin-top:11px}footer{text-align:center;color:#7183a0;font-size:12px;padding:22px} @media(max-width:600px){.shell{padding:24px 15px}.hero{margin-top:43px}.hero h1{font-size:34px}.badge{display:none}.searchbox{flex-wrap:wrap}.searchbox button{height:42px}.searchbox input{flex-basis:100%}.meta{flex-wrap:wrap}.score{margin-left:0;min-width:110px}}
</style></head><body><div class="shell"><header><div class="brand"><div class="mark">⌕</div>CodeLens</div><div class="badge">⌁ &nbsp; CPU-first retrieval &nbsp;·&nbsp; Version-aware</div></header><section class="hero"><div class="eyebrow">Understand your codebase</div><h1>Find the code behind<br>your question.</h1><p>Search across functions, files, and versions. No generated answers — just relevant code.</p></section><div class="searchbox"><input id="q" placeholder="e.g. How is input cleaned before the main function?" onkeydown="if(event.key==='Enter')run()"><button onclick="run()">Search&nbsp; →</button></div><div class="filters"><select id="revision" onchange="run()"><option value="">Latest indexed version</option><option value="all">All versions</option></select><select id="limit" onchange="run()"><option>5</option><option selected>10</option><option>20</option></select></div><div class="hint">Scores are relative to the top result in this query. A red 0 means no match; scores are not probabilities.</div><main class="content" id="results"><div class="empty">Index a source folder or Git repository, then ask a question to explore its code.</div></main><footer>Local search · Your source stays on your machine</footer></div><script>
async function init(){let d=await fetch('/api/revisions').then(r=>r.json());let s=document.querySelector('#revision');for(let x of d.revisions){let o=document.createElement('option');o.value=x;o.textContent=x.slice(0,10);s.append(o)}}
async function run(){let q=document.querySelector('#q').value.trim();if(!q)return;let rev=document.querySelector('#revision').value;let n=document.querySelector('#limit').value;let r=await fetch('/api/search?q='+encodeURIComponent(q)+'&revision='+encodeURIComponent(rev)+'&limit='+n);let d=await r.json();let root=document.querySelector('#results');root.innerHTML='';let h=document.createElement('div');h.className='resulthead';h.innerHTML=`<span>${d.results.length} ranked code blocks</span><span>${d.elapsed_ms} ms · ${d.revisions.length} version${d.revisions.length===1?'':'s'} searched</span>`;root.append(h);if(!d.results.length){root.innerHTML+='<div class="empty">No indexed code found. Run the index command and try again.</div>';return}let maxScore=Math.max(0,...d.results.map(x=>x.score));for(let x of d.results){let c=document.createElement('article');c.className='card';let shown=x.score<=0||maxScore<=0?0:Math.round(x.score/maxScore*50)/10;let pct=Math.round(shown*20);let tone=x.score<=0?'score-zero':shown>=2.5?'score-good':'score-low';let label=x.score<=0?'No match':shown>=2.5?'Good match':'Weak match';let meta=document.createElement('div');meta.className='meta';meta.innerHTML=`<span class="lang">${esc(x.language)}</span><span>${esc(x.revision.slice(0,10))}</span><span>lines ${x.start_line}–${x.end_line}</span><span class="score ${tone}" title="Relative to the top result in this query; not a calibrated probability">${shown.toFixed(1)} / 5 · ${pct}%<small>${label}</small></span>`;let p=document.createElement('div');p.className='path';p.textContent=x.path;let sy=document.createElement('div');sy.className='symbol';sy.textContent=x.kind+' · '+x.symbol;let pre=document.createElement('pre');pre.textContent=x.snippet;c.append(meta,p,sy,pre);if(x.versions?.length>1){let hist=document.createElement("details");hist.className="history";let summary=document.createElement("summary");summary.textContent=`Seen in ${x.versions.length} versions`;hist.append(summary);let list=document.createElement("ul");for(const v of x.versions){let li=document.createElement("li");li.textContent=`${v.revision.slice(0,10)} · score ${v.score}`;list.append(li)}hist.append(list);c.append(hist)}root.append(c)}}function esc(s){let d=document.createElement('div');d.textContent=s;return d.innerHTML}init();
</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    dbpath='.codelens/index.sqlite'
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=='/': body=PAGE.encode(); typ='text/html; charset=utf-8'
        elif u.path=='/api/revisions':
            db=db_open(self.dbpath); revs=[x[0] for x in db.execute('SELECT revision,MAX(id) FROM blocks GROUP BY revision ORDER BY MAX(id) DESC')]; db.close()
            body=json.dumps({'revisions':revs}).encode();typ='application/json'
        elif u.path=='/api/search':
            a=parse_qs(u.query); q=a.get('q',[''])[0]; rev=a.get('revision',[''])[0]; n=min(50,max(1,int(a.get('limit',['10'])[0]))); allv=rev=='all'; rev=None if rev in ('','all') else rev
            if rev is None and not allv:
                db=db_open(self.dbpath); latest=db.execute('SELECT revision FROM blocks ORDER BY id DESC LIMIT 1').fetchone(); db.close()
                rev=latest['revision'] if latest else None
            import time; st=time.perf_counter(); results=search(q,rev,n,self.dbpath,allv);elapsed=round((time.perf_counter()-st)*1000,1)
            db=db_open(self.dbpath); indexed=[row[0] for row in db.execute('SELECT DISTINCT revision FROM blocks ORDER BY revision')]; db.close(); searched=indexed if allv else ([rev] if rev else [])
            body=json.dumps({'results':results,'elapsed_ms':elapsed,'revisions':searched}).encode();typ='application/json'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args): pass

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    i=sub.add_parser('index');i.add_argument('repo');i.add_argument('--revision',default='HEAD');i.add_argument('--all',action='store_true');i.add_argument('--db',default='.codelens/index.sqlite')
    s=sub.add_parser('search');s.add_argument('query');s.add_argument('--revision');s.add_argument('--top-k',type=int,default=10);s.add_argument('--all-versions',action='store_true');s.add_argument('--db',default='.codelens/index.sqlite')
    v=sub.add_parser('serve');v.add_argument('--host',default='127.0.0.1');v.add_argument('--port',type=int,default=8765);v.add_argument('--db',default='.codelens/index.sqlite')
    a=p.parse_args()
    try:
        if a.cmd=='index':
            revs=git(a.repo,'rev-list','--reverse','HEAD').splitlines() if a.all else [a.revision]
            for rev in revs:
                started=time.perf_counter()
                commit,count,stats=index_revision(a.repo,rev,a.db);print(f'Indexed {count} blocks at {commit[:12]}')
                print(f"Index time: {time.perf_counter()-started:.2f} s | changed paths={stats['changed_paths']} | source files read={stats['files_read']} | embedding cache hits={stats['embedding_cache_hits']} misses={stats['embedding_cache_misses']}")
        elif a.cmd=='search': print(json.dumps(search(a.query,a.revision,a.top_k,a.db,a.all_versions),indent=2))
        else: Handler.dbpath=a.db; print(f'CodeLens running at http://{a.host}:{a.port}');ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
    except (subprocess.CalledProcessError,FileNotFoundError) as e: print(f'Unable to access repository or revision: {e}',file=sys.stderr);sys.exit(2)
if __name__=='__main__': main()
