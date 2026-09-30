"""Compare dense and hybrid CodeLens retrieval on the MTEB AppsRetrieval test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MTEB_CACHE',str(PROJECT_ROOT/'.cache'/'mteb'))
os.environ.setdefault('HF_HOME',str(PROJECT_ROOT/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_CACHE',str(PROJECT_ROOT/'.cache'/'huggingface'/'hub'))

import mteb
from datasets import load_dataset
from mteb.models.model_meta import ModelMeta
import codelens
from codelens import BM25Index,encode_texts,reciprocal_rank_fusion,tokens

ENCODE_BATCH_SIZE=int(os.getenv('CODELENS_ENCODE_BATCH_SIZE','64'))

BASE_OUTPUT=Path(os.getenv('CODELENS_RESULTS','appsretrieval_results.json'))
_CORPUS_VECTORS={}
_QUERY_VECTORS={}

def _document_windows(text,limit,overlap):
    if limit<=0: return [text]
    spans=list(re.finditer(r'\S+',text))
    if len(spans)<=limit: return [text]
    step=max(1,limit-max(0,min(overlap,limit-1)))
    chunks=[]
    for start in range(0,len(spans),step):
        end=min(start+limit,len(spans))
        chunks.append(text[spans[start].start():spans[end-1].end()])
        if end==len(spans): break
    return chunks


class AppsRetrievalRanker:
    """MTEB adapter; each mode corresponds to an actual CodeLens retrieval path."""
    def __init__(self,mode,chunk_tokens=128,chunk_overlap=24,keep_content_terms=True,query_chunk_tokens=128):
        self.mode=mode
        self.chunk_tokens=max(0,int(chunk_tokens)); self.chunk_overlap=max(0,int(chunk_overlap))
        self.keep_content_terms=bool(keep_content_terms); self.query_chunk_tokens=max(0,int(query_chunk_tokens))
        self.docs=[]
        self.query_search_latencies_ms=[]
        info=codelens.encoder_info() if mode!='bm25' else {'model':'CodeLens-BM25-control','dimension':0,'parameters':0,'max_tokens':256}
        self.model_id=info['model']
        slug=re.sub(r'[^A-Za-z0-9]+','-',str(info['model'])).strip('-')
        model_name='CodeLens-BM25-control' if mode=='bm25' else f'CodeLens-{slug}-{mode}'
        license_name={'Salesforce/codet5p-110m-embedding':'bsd-3-clause','microsoft/unixcoder-base':'apache-2.0'}.get(info['model'],'apache-2.0')
        self.meta=ModelMeta(loader=None,name=f'R1thanya/{model_name}',revision=f'rank-v7-content-tf-c{self.chunk_tokens}-o{self.chunk_overlap}-legacy{int(not self.keep_content_terms)}-q{self.query_chunk_tokens}',release_date='2026-09-27',
            languages=['eng-Latn'],n_parameters=info['parameters'],memory_usage_mb=None,max_tokens=info['max_tokens'],embed_dim=info['dimension'],
            license=license_name,open_weights=True,public_training_code='https://github.com/R1thanya/codelens-retrieval',
            public_training_data=None,framework=['Sentence Transformers'],similarity_fn_name='cosine',
            use_instructions=False,training_datasets=None,model_type=['sparse' if mode=='bm25' else 'dense'])

    @property
    def mteb_model_meta(self): return self.meta

    def index(self,corpus,*,task_metadata,hf_split,hf_subset,encode_kwargs,num_proc=None):
        self.docs=[]
        for row in corpus:
            title=row.get('title','') or ''
            body=row.get('text','') or ''
            self.docs.append({'id':str(row['id']),'title':title,'text':body,'partition':row.get('partition',''),
                'terms':tokens(f'{title}\n{body}')})
        self.chunks=[]; self.chunk_to_doc=[]
        for doc_index,doc in enumerate(self.docs):
            text=codelens.prepare_code(f"{doc['title']}\n{doc['text']}")
            pieces=_document_windows(text,self.chunk_tokens,self.chunk_overlap)
            for chunk_index,piece in enumerate(pieces):
                self.chunks.append({'id':f"{doc['id']}#chunk-{chunk_index}",'text':piece,'terms':tokens(piece)})
                self.chunk_to_doc.append(doc_index)
        self.bm25=BM25Index(self.chunks) if self.mode in ('bm25','hybrid') else None
        if self.mode in ('dense','hybrid'):
            cache_key=(codelens.embedding_fingerprint(),self.chunk_tokens,self.chunk_overlap,
                tuple((c['id'],hashlib.sha256(c['text'].encode('utf-8')).hexdigest()) for c in self.chunks))
            if cache_key not in _CORPUS_VECTORS:
                _CORPUS_VECTORS[cache_key]=encode_texts([c['text'] for c in self.chunks],batch_size=ENCODE_BATCH_SIZE,role='document')
            self.vectors=_CORPUS_VECTORS[cache_key]

    def search(self,queries,*,task_metadata,hf_split,hf_subset,top_k,encode_kwargs,top_ranked=None,num_proc=None):
        from mteb._create_dataloaders import _combine_queries_with_instruction_text
        ids=list(queries['id']); texts=list(_combine_queries_with_instruction_text(queries)['text'])
        query_key=tuple(ids)
        if self.mode in ('dense','hybrid'):
            query_texts=[]; query_ranges=[]
            for query in texts:
                prepared=codelens.prepare_query(str(query))
                pieces=_document_windows(prepared,self.query_chunk_tokens,0)
                start=len(query_texts); query_texts.extend(pieces); query_ranges.append((start,len(query_texts)))
            query_key=(codelens.embedding_fingerprint(),self.query_chunk_tokens,tuple(ids))
            if query_key not in _QUERY_VECTORS: _QUERY_VECTORS[query_key]=encode_texts(query_texts,batch_size=ENCODE_BATCH_SIZE,role='query')
            qvectors=_QUERY_VECTORS[query_key]
        else: qvectors=None; query_ranges=[]
        import numpy as np
        output={}
        for row,(query_id,query) in enumerate(zip(ids,texts,strict=True)):
            query_started=time.perf_counter()
            if self.mode=='bm25':
                chunk_scores=self.bm25.scores(str(query)); scores=np.zeros(len(self.docs),dtype=np.float32)
                np.maximum.at(scores,self.chunk_to_doc,np.asarray(chunk_scores,dtype=np.float32))
            else:
                query_start,query_end=query_ranges[row]
                chunk_dense=np.max(self.vectors@qvectors[query_start:query_end].T,axis=1)
                dense=np.full(len(self.docs),-np.inf,dtype=np.float32)
                np.maximum.at(dense,self.chunk_to_doc,chunk_dense)
                if self.mode=='dense': scores=dense
                else:
                    chunk_lexical=self.bm25.scores(str(query)); lexical=np.zeros(len(self.docs),dtype=np.float32)
                    np.maximum.at(lexical,self.chunk_to_doc,np.asarray(chunk_lexical,dtype=np.float32))
                    scores=codelens.fuse_scores(lexical,dense)
            values=np.asarray(scores,dtype=np.float32)
            candidates=np.flatnonzero(values>0) if self.mode=='bm25' else np.arange(len(self.docs))
            if top_ranked and query_id in top_ranked:
                allowed=set(top_ranked[query_id]); candidates=np.asarray([i for i in candidates if self.docs[i]['id'] in allowed],dtype=np.int64)
            if len(candidates)>top_k:
                part=np.argpartition(-values[candidates],top_k-1)[:top_k]
                candidates=candidates[part]
            ranked=sorted(candidates.tolist(),key=lambda i:(-scores[i],self.docs[i]['id']))
            output[query_id]={self.docs[i]['id']:float(scores[i]) for i in ranked[:top_k]}
            self.query_search_latencies_ms.append((time.perf_counter()-query_started)*1000)
        return output


def _json_default(value):
    if isinstance(value,(datetime,date)): return value.isoformat()
    if hasattr(value,'item'): return value.item()
    if hasattr(value,'value'): return value.value
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def load_apps_from_revision_cache(task):
    """Load the pinned CoIR configs via datasets' cache, then hand them to MTEB.

    MTEB 2.21.8's generic loader probes the Hub's dataset script even when all
    parquet configs are cached. CoIR's configs are data-only, so construct the
    same MTEB retrieval split from its pinned qrels/queries/corpus configs.
    """
    rev=task.metadata.dataset['revision']
    repo=task.metadata.dataset['path']
    qrels_ds=load_dataset(repo,'default',revision=rev)['test']
    queries_ds=load_dataset(repo,'queries',revision=rev)['queries']
    corpus_ds=load_dataset(repo,'corpus',revision=rev)['corpus']
    qrels={}
    for row in qrels_ds:
        qrels.setdefault(str(row['query-id']),{})[str(row['corpus-id'])]=int(row['score'])
    from datasets import Value
    for ds_name,ds in [('queries',queries_ds),('corpus',corpus_ds)]:
        if '_id' in ds.column_names:
            ds=ds.cast_column('_id',Value('string')).rename_column('_id','id')
            if ds_name=='queries': queries_ds=ds
            else: corpus_ds=ds
    ids=set(qrels)
    queries_ds=queries_ds.filter(lambda row:str(row['id']) in ids)
    task.dataset={'default':{'test':{'queries':queries_ds,'corpus':corpus_ds,
        'relevant_docs':qrels,'top_ranked':None}}}
    task.data_loaded=True
    print(f'Loaded pinned CoIR revision {rev}: {len(queries_ds)} test queries, {len(corpus_ds)} corpus documents.')


def evaluate(mode,task,out,tune_train=False,fusion_config=None,chunk_tokens=128,chunk_overlap=24,keep_content_terms=True,query_chunk_tokens=128):
    started=time.perf_counter()
    if fusion_config: codelens.FUSION_CONFIG_FILE=Path(fusion_config)
    ranker=AppsRetrievalRanker(mode,chunk_tokens,chunk_overlap,keep_content_terms,query_chunk_tokens)
    initialized=time.perf_counter()
    result=mteb.evaluate(ranker,[task],encode_kwargs={'batch_size':ENCODE_BATCH_SIZE})
    elapsed=time.perf_counter()-started
    payload=list(result.task_results)[0].to_dict()
    rows=payload.get('scores',{}).get('test',[])
    if not rows or 'ndcg_at_10' not in rows[0] or 'mrr_at_10' not in rows[0]:
        raise RuntimeError(f'MTEB {mode} result did not include AppsRetrieval test NDCG@10 and MRR@10')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=_json_default)+'\n',encoding='utf-8')
    print(f'{mode.upper():6} NDCG@10={rows[0]["ndcg_at_10"]:.5f} MRR@10={rows[0]["mrr_at_10"]:.5f} | {out.resolve()}')
    info=codelens.encoder_info() if mode!='bm25' else {'model':'CodeLens-BM25-control','dimension':0,'parameters':0,'max_tokens':0}
    import numpy as np
    latencies=ranker.query_search_latencies_ms
    timing={'mode':mode,'model':info,'mteb_test_split':'test','total_wall_seconds':elapsed,
        'query_search_component_p50_ms':float(np.percentile(latencies,50)) if latencies else None,
        'query_search_component_p95_ms':float(np.percentile(latencies,95)) if latencies else None,
        'query_search_component_queries':len(latencies),
        'initialization_wall_seconds':initialized-started,'model_load_seconds':codelens.ENCODER_TIMINGS['model_load_seconds'],
        'corpus_encode_wall_seconds':codelens.ENCODER_TIMINGS['document_encode_seconds'],
        'query_encode_wall_seconds':codelens.ENCODER_TIMINGS['query_encode_seconds'],
        'documents_encoded':codelens.ENCODER_TIMINGS['documents_encoded'],'queries_encoded':codelens.ENCODER_TIMINGS['queries_encoded']}
    out.with_name(out.stem+'_timing.json').write_text(json.dumps(timing,indent=2)+'\n',encoding='utf-8')
    print('TIMING '+json.dumps(timing))
    return rows[0]


def main():
    global BASE_OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['all','bm25','dense','hybrid'],default='all')
    parser.add_argument('--model',help='Sentence-Transformers or supported code encoder model id (default: current MiniLM encoder)')
    parser.add_argument('--tune-train',action='store_true',help='Tune fusion on a held-out AppsRetrieval train sample; never uses test qrels')
    parser.add_argument('--tune-train-queries',type=int,default=500,help='Maximum held-out training queries for --tune-train')
    parser.add_argument('--tune-train-documents',type=int,default=1500,help='Maximum training corpus documents for --tune-train; includes sampled positives')
    parser.add_argument('--fusion-config',help='JSON fusion configuration for hybrid evaluation')
    parser.add_argument('--chunk-tokens',type=int,default=128,help='Maximum whitespace tokens per code window (default: 128)')
    parser.add_argument('--chunk-overlap',type=int,default=24,help='Overlap size for corpus windows (default: 24)')
    parser.add_argument('--query-chunk-tokens',type=int,default=128,help='Query window size; code similarity takes the maximum across windows (default: 128)')
    parser.add_argument('--drop-content-terms',action='store_false',dest='keep_content_terms',default=True,help='Restore the legacy stop list that drops input/output/data/value/code/file/function/main')
    args=parser.parse_args()
    if args.tune_train and args.mode!='dense': parser.error('--tune-train requires --mode dense')
    if args.model: codelens.set_embedding_model(args.model)
    if not args.keep_content_terms:
        codelens.STOP.update({'input','output','data','value','code','file','function','main'})
    if args.model and not os.getenv('CODELENS_RESULTS') and BASE_OUTPUT.name=='appsretrieval_results.json':
        slug=re.sub(r'[^A-Za-z0-9]+','-',args.model).strip('-').lower()
        BASE_OUTPUT=BASE_OUTPUT.with_name(f'appsretrieval_{slug}_results.json')
    if args.tune_train:
        from fusion_tuning import tune_train_validation
        slug=re.sub(r'[^A-Za-z0-9]+','-',codelens.EMBEDDING_MODEL).strip('-').lower()
        tuning_path=BASE_OUTPUT.with_name(f'fusion_tuning_{slug}.json')
        tuned=tune_train_validation(tuning_path,batch_size=ENCODE_BATCH_SIZE,query_limit=args.tune_train_queries,document_limit=args.tune_train_documents)
        config_path=BASE_OUTPUT.with_name(f'fusion_{slug}.json')
        config_path.write_text(json.dumps(tuned['best_config']['config'],indent=2)+'\n',encoding='utf-8')
        print(f'TRAIN-VAL best={json.dumps(tuned["best_config"])}')
        print(f'Wrote {tuning_path.resolve()} and {config_path.resolve()}')
        return
    task=mteb.get_task('AppsRetrieval')
    load_apps_from_revision_cache(task)
    modes=['bm25','dense','hybrid'] if args.mode=='all' else [args.mode]
    scores={}
    for mode in modes:
        if args.mode=='all' and mode!='hybrid': out=BASE_OUTPUT.with_name(f'{BASE_OUTPUT.stem}_{mode}{BASE_OUTPUT.suffix}')
        elif os.getenv('CODELENS_RESULTS') or mode=='hybrid': out=BASE_OUTPUT
        else: out=BASE_OUTPUT.with_name(f'{BASE_OUTPUT.stem}_{mode}{BASE_OUTPUT.suffix}')
        if (args.chunk_tokens,args.chunk_overlap,args.query_chunk_tokens)!=(128,24,128):
            if args.chunk_tokens>0: out=out.with_name(f'{out.stem}_chunks{args.chunk_tokens}_o{args.chunk_overlap}{out.suffix}')
            else: out=out.with_name(f'{out.stem}_unchunked{out.suffix}')
            if args.query_chunk_tokens>0: out=out.with_name(f'{out.stem}_qchunks{args.query_chunk_tokens}{out.suffix}')
            else: out=out.with_name(f'{out.stem}_wholequery{out.suffix}')
        scores[mode]=evaluate(mode,task,out,False,args.fusion_config,args.chunk_tokens,args.chunk_overlap,args.keep_content_terms,args.query_chunk_tokens)
    if args.mode=='all':
        baseline=scores['bm25']['ndcg_at_10']; hybrid=scores['hybrid']['ndcg_at_10']
        print(f'Hybrid NDCG@10 change vs plain BM25: {hybrid-baseline:+.5f} ({(hybrid/baseline-1)*100:+.1f}%)')


if __name__=='__main__': main()
