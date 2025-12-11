"""
RAG Engine - Core logic extracted from rag.ipynb
"""

import gc
import re
import unicodedata
import torch
from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


class RAGEngine:
    """RAG Engine for biomedical article summarization"""
    
    def __init__(self, checkpoint="Qwen/Qwen2.5-3B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing RAG Engine on {self.device}...")
        
        # Config
        self.max_new_tokens = 512
        self.num_beams = 2
        self.input_max_length = 4096
        
        # Load Qwen model
        print("  Loading Qwen model...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        # Load embedding model
        print("  Loading embedding model...")
        self.embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Load cross-encoder
        print("  Loading cross-encoder...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device)
        
        print("✓ RAG Engine initialized!\n")
    
    def free_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def summary_length_from_tokens(self, n_tokens: int) -> str:
        if n_tokens <= 5000:
            return "100-300 words"
        elif n_tokens <= 8000:
            return "150-400 words"
        elif n_tokens <= 12000:
            return "250-600 words"
        else:
            return "300-700 words"
    
    def get_summary_length(self, sample) -> str:
        n_tokens = sample.get("n_tokens", None)
        if n_tokens is None:
            text = sample.get("main_text_clean") or ""
            n_tokens = len(text.split())
        return self.summary_length_from_tokens(int(n_tokens))
    
    def normalize_article(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r'\.\s{2,}([A-Z])', r'.\n\1', text)
        return text
    
    def clean_fulltext(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"(Figure|Fig\.?|Table)\s*\d+[^.\n]*[.\n]", " ", text, flags=re.IGNORECASE)
        citation_pattern = r"\[\s*(?:\d+\s*(?:[-–]\s*\d+)?\s*(?:,\s*)?)+\]"
        text = re.sub(citation_pattern, " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def extract_abstract(self, sample: dict) -> dict:
        article = sample.get("article", "") or ""
        text = article.strip()
        if not text:
            sample["abstract"] = ""
            sample["main_text"] = ""
            return sample
        
        first_line, sep, rest = text.partition("\n")
        
        if not sep:
            sample["abstract"] = text
            sample["main_text"] = ""
        else:
            sample["abstract"] = first_line.strip()
            sample["main_text"] = rest.strip()
        
        return sample
    
    def preprocess(self, sample: dict) -> dict:
        main_text = sample.get("main_text", "") or ""
        abstract = sample.get("abstract", "") or ""
        sample["main_text_clean"] = self.clean_fulltext(main_text)
        sample["abstract"] = abstract.strip()
        return sample
    
    def choose_chunk_params(self, main_text_clean: str):
        if not main_text_clean:
            return 0, 0, 0, 0
        n_tokens = len(main_text_clean.split())
        if n_tokens <= 5000:
            return n_tokens, 900, 120, 4
        elif n_tokens <= 8000:
            return n_tokens, 700, 100, 6
        elif n_tokens <= 12000:
            return n_tokens, 550, 80, 8
        else:
            return n_tokens, 450, 60, 10
    
    def add_chunks(self, example):
        text = example.get("main_text_clean", "")
        if not text:
            example["main_text_chunks"] = []
            example["n_tokens"] = 0
            example["k_final"] = 0
            return example
        
        n_tokens, chunk_size, chunk_overlap, k_final = self.choose_chunk_params(text)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        chunks = splitter.split_text(text)
        
        example["main_text_chunks"] = chunks
        example["n_tokens"] = n_tokens
        example["k_final"] = k_final
        return example
    
    def build_queries(self, abstract: str):
        if not abstract:
            return []
        sents = [s.strip() for s in sent_tokenize(abstract) if s.strip()]
        return sents if sents else [abstract]
    
    def retrieve_chunks(self, example, top_k_per_query: int = 3, return_debug: bool = False):
        chunks = example.get("main_text_chunks", [])
        abstract = example.get("abstract", "")
        k_final = example.get("k_final", 0)
        main_text = example.get("main_text", "")
        
        if not chunks or not abstract or k_final <= 0:
            example["retrieved_context"] = ""
            if return_debug:
                example["rag_debug"] = {"steps": []}
            return example
        
        candidate_cap = min(len(chunks), k_final * 3)
        queries = self.build_queries(abstract)
        if not queries:
            queries = [abstract]
        
        docsearch = FAISS.from_texts(chunks, self.embedder)
        
        # Track debug info
        debug_steps = []
        query_details = []
        
        # Dense retrieval - track which queries select which chunks
        scores = defaultdict(float)
        chunk_query_map = defaultdict(list)  # Track which queries selected each chunk
        total_retrievals = 0
        
        for q_idx, q in enumerate(queries):
            docs = docsearch.similarity_search(q, k=top_k_per_query)
            query_chunks = []
            for rank, d in enumerate(docs):
                chunk_text = d.page_content
                scores[chunk_text] += (top_k_per_query - rank)
                chunk_query_map[chunk_text].append(q_idx + 1)  # Track query number
                total_retrievals += 1
                query_chunks.append({
                    "chunk_text": chunk_text,
                    "rank": rank + 1,
                    "score": top_k_per_query - rank
                })
            query_details.append({
                "query": q[:100] + "..." if len(q) > 100 else q,
                "retrieved_chunks": query_chunks
            })
        
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidate_texts = [t for t, _ in sorted_candidates[:candidate_cap]]
        
        # Find chunk positions in original text
        def find_chunk_position(chunk, text):
            """Find approximate position of chunk in main text"""
            # Clean both for matching
            clean_chunk = ' '.join(chunk.split()[:20])  # First 20 words
            clean_text = text.lower()
            clean_chunk_search = clean_chunk.lower()
            
            # Try to find position
            pos = clean_text.find(clean_chunk_search)
            if pos != -1:
                # Count words before this position
                words_before = len(clean_text[:pos].split())
                return words_before
            return -1
        
        dense_results = []
        for i, (text, score) in enumerate(sorted_candidates[:candidate_cap]):
            pos = find_chunk_position(text, main_text)
            selected_by_queries = chunk_query_map.get(text, [])
            dense_results.append({
                "chunk_text": text,
                "rank": i + 1,
                "score": float(score),
                "position": pos,
                "length": len(text.split()),
                "selected_by_queries": selected_by_queries,
                "query_count": len(selected_by_queries)
            })
        
        # Rerank
        pairs = [[abstract, ctx] for ctx in candidate_texts]
        rerank_scores = self.cross_encoder.predict(pairs)
        
        reranked = sorted(zip(candidate_texts, rerank_scores), key=lambda x: x[1], reverse=True)
        
        rerank_results = []
        for i, (text, score) in enumerate(reranked):
            pos = find_chunk_position(text, main_text)
            selected_by_queries = chunk_query_map.get(text, [])
            rerank_results.append({
                "chunk_text": text,
                "rank": i + 1,
                "rerank_score": float(score),
                "position": pos,
                "selected": i < k_final,
                "length": len(text.split()),
                "selected_by_queries": selected_by_queries,
                "query_count": len(selected_by_queries)
            })
        
        selected_texts = [t for t, _ in reranked[:k_final]]
        example["retrieved_context"] = "\n\n".join(selected_texts)
        
        if return_debug:
            example["rag_debug"] = {
                "num_queries": len(queries),
                "queries": [q[:80] + "..." if len(q) > 80 else q for q in queries],
                "num_total_chunks": len(chunks),
                "total_retrievals": total_retrievals,  # 24 (before dedup)
                "unique_candidates": len(sorted_candidates),  # 12 (after dedup)
                "dense_candidates": dense_results,
                "reranked_results": rerank_results,
                "final_chunks": rerank_results[:k_final],
                "deduplication_info": {
                    "before": total_retrievals,
                    "after": len(candidate_texts),
                    "removed": total_retrievals - len(candidate_texts)
                }
            }
        
        return example
    
    def build_prompt(self, sample, version: str) -> str:
        summary_word_len = self.get_summary_length(sample)
        
        prompts = {
            "formal": f"""<|im_start|>system
You are an expert biomedical science communicator.
Generate a *formal, scientifically rigorous* summary grounded strictly in the evidence provided.
The writing style should resemble academic press releases or journal lay summaries.
Organize the summary with clear conceptual sections — such as:
- Background / Rationale
- Methods or Study Approach
- Key Findings
- Implications or Significance
Do NOT output headings; instead, write structured paragraphs that implicitly follow this organization.
Use precise technical vocabulary, complex but grammatically correct sentences, and objective tone.
Avoid oversimplification.<|im_end|>
<|im_start|>user
Title: {sample['title']}
Abstract: {sample['abstract']}

Supporting Text:
{sample['retrieved_context']}

Write a *formal scientific lay summary* of approximately {summary_word_len} words.
Begin directly with the summary text.<|im_end|>
<|im_start|>assistant
""",
            "plain": f"""<|im_start|>system
You are an expert biomedical science communicator.
Generate a *clear, accessible, and accurate* plain-language summary.
Organize the text: what the study is about, how it was done, what was found, why it matters.
Use short-to-medium sentences and everyday vocabulary.
Explain technical terms concisely.<|im_end|>
<|im_start|>user
Title: {sample['title']}
Abstract: {sample['abstract']}

Supporting Text:
{sample['retrieved_context']}

Write an *accessible plain-language summary* of approximately {summary_word_len} words.
Begin directly with the summary text.<|im_end|>
<|im_start|>assistant
""",
            "high_readability": f"""<|im_start|>system
You are an expert science communicator.
Generate a *highly readable* summary for readers with no biomedical background.
Use short sentences (10-14 words), common vocabulary, simple grammar.
Explain all technical ideas directly.
Avoid jargon and dense details.<|im_end|>
<|im_start|>user
Title: {sample['title']}
Abstract: {sample['abstract']}

Supporting Text:
{sample['retrieved_context']}

Write a *high-readability lay summary* of approximately {summary_word_len} words.
Begin directly with the summary text.<|im_end|>
<|im_start|>assistant
"""
        }
        
        return prompts.get(version, prompts["plain"])
    
    def generate_summary(self, title: str, article: str, version: str = "plain", include_debug: bool = True):
        """Main processing pipeline with optional debug info"""
        
        if not article.strip():
            raise ValueError("Article cannot be empty")
        
        if not title.strip():
            title = "Untitled Article"
        
        # Build sample
        norm_article = self.normalize_article(article)
        sample = {"article": norm_article, "title": title}
        
        # Extract and preprocess
        sample = self.extract_abstract(sample)
        if not sample.get("main_text"):
            sample["main_text"] = sample.get("article", "")
        sample = self.preprocess(sample)
        
        # Chunking and retrieval (with debug info)
        sample = self.add_chunks(sample)
        sample = self.retrieve_chunks(sample, return_debug=include_debug)
        
        # Generate
        prompt = self.build_prompt(sample, version=version)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.input_max_length,
        )
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract summary
        if "<|im_start|>assistant" in decoded:
            summary = decoded.split("<|im_start|>assistant", -1)[-1].strip()
        elif "assistant\n" in decoded:
            summary = decoded.split("assistant\n", 1)[-1].strip()
        else:
            summary = decoded.strip()
        
        # Clean cache
        self.free_cuda()
        
        result = {
            "summary": summary,
            "word_count": len(summary.split()),
            "chunks_used": len(sample.get("main_text_chunks", [])),
            "queries": len(self.build_queries(sample.get("abstract", ""))),
            "abstract": sample.get("abstract", ""),
            "main_text": sample.get("main_text", ""),
        }
        
        # Add debug info if requested
        if include_debug and "rag_debug" in sample:
            result["rag_process"] = sample["rag_debug"]
        
        return result

