"""Tune hybrid settings on a deterministic held-out slice of AppsRetrieval train."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import mteb
from datasets import load_dataset
import codelens
from codelens import BM25Index, encode_texts

DATASET_REVISION='f22508f96b7a36c2415181ed8bb76f76e04ae2d5'

def tune_train_validation(output_path,seed=20260927,holdout_fraction=.20,batch_size=64,query_limit=500,document_limit=1500):
    # Work only on the training partition: 5,000 train documents and a held-out
    # query slice. This avoids encoding the unrelated test corpus before tuning.
    mteb.get_task('AppsRetrieval')  # validate task name exists in installed MTEB
    raw_docs=load_dataset('CoIR-Retrieval/apps','corpus',revision=DATASET_REVISION)['corpus']
    all_docs=[row for row in raw_docs if row['partition']=='train']
    train_docs=[{'id':str(row['_id']),'title':row.get('title','') or '',
        'text':row.get('text','') or '', 'terms':codelens.tokens(f"{row.get('title','')}\n{row.get('text','')}"),
        'partition':'train'} for row in all_docs]
    docs_ids=[d['id'] for d in train_docs]
    id_to_position={d:i for i,d in enumerate(docs_ids)}
    qrels=load_dataset('CoIR-Retrieval/apps',revision=DATASET_REVISION)['train']
    queries=load_dataset('CoIR-Retrieval/apps','queries',revision=DATASET_REVISION)['queries']
    query_by_id={str(row['_id']):row for row in queries if row['partition']=='train'}
    positives={}
    for row in qrels:
        qid=str(row['query-id']); did=str(row['corpus-id'])
        if qid in query_by_id and did in id_to_position:
            positives[qid]=id_to_position[did]
    if len(positives)<4000:
        raise RuntimeError(f'Only {len(positives)} train query/code labels map into the training subset')
    # Stratified deterministic holdout; the current set is Python-only, but this also
    # remains stable if language labels are added later.
    groups={}
    for qid in positives:
        language=str(query_by_id[qid].get('language','unknown'))
        groups.setdefault(language,[]).append(qid)
    val_ids=[]
    for language,ids in sorted(groups.items()):
        ranked=sorted(ids,key=lambda q:(hashlib.sha256(f'{seed}:{language}:{q}'.encode()).hexdigest(),q))
        val_ids.extend(ranked[:max(1,round(len(ranked)*holdout_fraction))])
    val_ids=sorted(val_ids,key=lambda q:(hashlib.sha256(f'{seed}:validation:{q}'.encode()).hexdigest(),q))
    if query_limit>0: val_ids=val_ids[:query_limit]
    # Keep every sampled query's positive code document, then add a stable sample
    # of negatives up to the configured corpus budget.
    if document_limit>0 and len(train_docs)>document_limit:
        positive_id_by_query={qid:docs_ids[positives[qid]] for qid in val_ids}
        positive_doc_ids=set(positive_id_by_query.values())
        chosen=set(positive_doc_ids)
        remaining=sorted((did for did in docs_ids if did not in chosen),key=lambda did:(hashlib.sha256(f'{seed}:document:{did}'.encode()).hexdigest(),did))
        chosen.update(remaining[:max(0,document_limit-len(chosen))])
        train_docs=[d for d in train_docs if d['id'] in chosen]
        docs_ids=[d['id'] for d in train_docs]
        id_to_position={did:i for i,did in enumerate(docs_ids)}
        positives={qid:id_to_position[positive_id_by_query[qid]] for qid in val_ids}
    start=time.perf_counter()
    doc_vectors=encode_texts([codelens.prepare_code(f"{d['title']}\n{d['text']}") for d in train_docs],batch_size=batch_size,role='document')
    document_encode_seconds=time.perf_counter()-start
    start=time.perf_counter()
    query_texts=[codelens.prepare_query(str(query_by_id[qid]['text'])) for qid in val_ids]
    qvectors=encode_texts(query_texts,batch_size=batch_size,role='query')
    val_encode_seconds=time.perf_counter()-start
    dense=qvectors @ doc_vectors.T
    bm25_index=BM25Index(train_docs)
    bm25=np.asarray([bm25_index.scores(str(query_by_id[qid]['text'])) for qid in val_ids],dtype=np.float32)
    relevant=np.asarray([positives[qid] for qid in val_ids],dtype=np.int64)
    doc_ids=np.asarray(docs_ids,dtype=object)

    def metrics(scores):
        # Each Apps train query has one binary relevant-code judgement.
        rows=np.arange(len(relevant)); target=scores[rows,relevant]
        better=(scores>target[:,None]) | ((scores==target[:,None]) & (doc_ids[None,:]<doc_ids[relevant,None]))
        ranks=1+better.sum(axis=1)
        reciprocal=np.where(ranks<=10,1/ranks,0.0)
        ndcg=np.where(ranks<=10,1/np.log2(ranks+1),0.0)
        return {'ndcg_at_10':float(ndcg.mean()),'mrr_at_10':float(reciprocal.mean())}

    def ranked_rows(scores):
        ranks=np.empty(scores.shape,dtype=np.int32)
        ordinal=np.arange(scores.shape[1],dtype=np.int32)
        for i,row in enumerate(scores):
            order=np.lexsort((doc_ids,-row))
            ranks[i,order]=ordinal+1
        return ranks
    lexical_ranks=ranked_rows(bm25)
    semantic_ranks=ranked_rows(dense)
    bmin=bm25.min(axis=1,keepdims=True); bmax=bm25.max(axis=1,keepdims=True)
    smin=dense.min(axis=1,keepdims=True); smax=dense.max(axis=1,keepdims=True)
    bn=np.divide(bm25-bmin,bmax-bmin,out=np.zeros_like(bm25),where=(bmax>bmin))
    sn=np.divide(dense-smin,smax-smin,out=np.zeros_like(dense),where=(smax>smin))
    configs=[]
    cutoffs=[None,0.0,0.1,0.2,0.3,0.4,0.5]
    for k in [10,20,40,60,100]:
        for weight in [0.25,0.5,0.75]:
            for cutoff in cutoffs:
                configs.append({'strategy':'rrf','rrf_k':k,'bm25_weight':weight,'semantic_weight':1-weight,'semantic_cutoff':cutoff})
    for weight in [0.25,0.5,0.75]:
        for cutoff in cutoffs:
            configs.append({'strategy':'minmax','rrf_k':None,'bm25_weight':weight,'semantic_weight':1-weight,'semantic_cutoff':cutoff})
    runs=[]
    for config in configs:
        if config['strategy']=='rrf':
            k=config['rrf_k']; w=config['bm25_weight']
            scores=np.where(bm25>0,w/(k+lexical_ranks),0.0)+(1-w)/(k+semantic_ranks)
        else:
            w=config['bm25_weight']
            scores=w*bn+(1-w)*sn
        cutoff=config['semantic_cutoff']
        if cutoff is not None: scores=np.where((bm25<=0)&(dense<float(cutoff)),0.0,scores)
        runs.append({'config':config,**metrics(scores)})
    best=max(runs,key=lambda x:(x['ndcg_at_10'],x['mrr_at_10']))
    baseline={'strategy':'rrf','rrf_k':60,'bm25_weight':.5,'semantic_weight':.5,'semantic_cutoff':None}
    current={'config':baseline,**metrics(np.where(bm25>0,.5/(60+lexical_ranks),0.0)+.5/(60+semantic_ranks))}
    payload={'dataset':'CoIR-Retrieval/apps','dataset_revision':DATASET_REVISION,'source_split':'train',
        'split_method':'stable SHA-256 ordering stratified by query language','seed':seed,
        'holdout_fraction':holdout_fraction,'validation_queries':len(val_ids),'train_pairs':len(positives),
        'validation_documents':len(train_docs),'query_limit':query_limit,'document_limit':document_limit,'encoder':codelens.encoder_info(),
        'document_encode_seconds':document_encode_seconds,'query_encode_seconds':val_encode_seconds,'current_equal_rrf':current,
        'best_config':best,'configs_evaluated':len(runs),'grid':runs}
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    return payload
