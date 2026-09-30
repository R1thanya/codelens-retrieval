"""CPU embedding adapters for the encoders evaluated by CodeLens."""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Large default thread pools can severely oversubscribe small Windows CPU hosts.
torch.set_num_threads(max(1,int(__import__('os').environ.get('CODELENS_TORCH_THREADS','4'))))


def _model_path(model_id: str, cache_dir: str | None) -> str:
    candidate=Path(model_id)
    if candidate.exists():
        return str(candidate)
    if cache_dir and '/' in model_id:
        hub=Path(cache_dir)
        repo='models--'+model_id.replace('/','--')
        refs=hub/repo/'refs'/'main'
        if refs.exists():
            snapshot=hub/repo/'snapshots'/refs.read_text(encoding='utf-8').strip()
            if (snapshot/'config.json').exists():
                return str(snapshot)
    return model_id


class SentenceTransformerAdapter:
    def __init__(self, model_id: str, cache_dir: str | None):
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(_model_path(model_id,cache_dir),cache_folder=cache_dir,device='cpu')
        self.dimension=int(self.model.get_sentence_embedding_dimension())
        self.parameters=sum(p.numel() for p in self.model.parameters())
        self.max_tokens=int(self.model.max_seq_length)

    def encode(self, texts, batch_size: int):
        return self.model.encode(texts,convert_to_numpy=True,normalize_embeddings=True,
            show_progress_bar=False,batch_size=batch_size).astype('float32')


class CodeT5pEmbeddingAdapter:
    """Adapter around Salesforce's published custom CodeT5+ embedding head."""
    def __init__(self, model_id: str, cache_dir: str | None):
        from transformers import AutoModel
        path=_model_path(model_id,cache_dir)
        self.tokenizer=AutoTokenizer.from_pretrained(path,cache_dir=cache_dir,trust_remote_code=True)
        self.model=AutoModel.from_pretrained(path,cache_dir=cache_dir,trust_remote_code=True).to('cpu').eval()
        self.dimension=int(self.model.config.embed_dim)
        self.parameters=sum(p.numel() for p in self.model.parameters())
        self.max_tokens=256

    @torch.inference_mode()
    def encode(self, texts, batch_size: int):
        outputs=[]
        for start in range(0,len(texts),batch_size):
            batch=self.tokenizer(texts[start:start+batch_size],padding=True,truncation=True,
                max_length=self.max_tokens,return_tensors='pt')
            vectors=self.model(**batch)
            if isinstance(vectors,(tuple,list)):
                vectors=vectors[0]
            outputs.append(F.normalize(vectors.float(),p=2,dim=-1).cpu())
        return torch.cat(outputs).numpy().astype('float32') if outputs else torch.empty((0,self.dimension)).numpy()


class UniXcoderAdapter:
    """Microsoft's encoder-only UniXcoder representation with mean pooling."""
    def __init__(self, model_id: str, cache_dir: str | None):
        path=_model_path(model_id,cache_dir)
        self.tokenizer=AutoTokenizer.from_pretrained(path,cache_dir=cache_dir)
        self.model=AutoModel.from_pretrained(path,cache_dir=cache_dir).to('cpu').eval()
        self.dimension=int(self.model.config.hidden_size)
        self.parameters=sum(p.numel() for p in self.model.parameters())
        self.max_tokens=256
        self.mode_id=self.tokenizer.convert_tokens_to_ids('<encoder-only>')
        if self.mode_id is None or self.mode_id==self.tokenizer.unk_token_id:
            raise RuntimeError('UniXcoder tokenizer is missing its <encoder-only> mode token')

    @torch.inference_mode()
    def encode(self, texts, batch_size: int):
        outputs=[]
        tok=self.tokenizer
        for start in range(0,len(texts),batch_size):
            rows=[]
            for text in texts[start:start+batch_size]:
                ids=tok.encode(str(text),add_special_tokens=False)[:self.max_tokens-4]
                rows.append([tok.cls_token_id,self.mode_id,tok.sep_token_id,*ids,tok.sep_token_id])
            width=max(map(len,rows))
            input_ids=torch.full((len(rows),width),tok.pad_token_id,dtype=torch.long)
            mask=torch.zeros_like(input_ids)
            for i,row in enumerate(rows):
                input_ids[i,:len(row)]=torch.tensor(row,dtype=torch.long)
                mask[i,:len(row)]=1
            hidden=self.model(input_ids=input_ids,attention_mask=mask).last_hidden_state
            pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            outputs.append(F.normalize(pooled.float(),p=2,dim=-1).cpu())
        return torch.cat(outputs).numpy().astype('float32') if outputs else torch.empty((0,self.dimension)).numpy()


def load_encoder(model_id: str, cache_dir: str | None):
    if model_id=='Salesforce/codet5p-110m-embedding':
        return CodeT5pEmbeddingAdapter(model_id,cache_dir)
    if model_id=='microsoft/unixcoder-base':
        return UniXcoderAdapter(model_id,cache_dir)
    return SentenceTransformerAdapter(model_id,cache_dir)
