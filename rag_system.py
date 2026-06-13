import os
import json
import logging
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types as genai_types

# ================================================================
# 0. Logging & Config & API KEY
# ================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Advanced_IDS_RAG")

# ---------------------------------------------------------
# ضع مفتاح Gemini الخاص بك هنا (تستطيع تغييره في أي وقت بسهولة)
# ---------------------------------------------------------
GEMINI_API_KEY = "AQ.Ab8RN6L5E2H4-XP6V619IYiWMiaS5yjg1Gb8C4-HRUODeuQn7Q" 

if not GEMINI_API_KEY or GEMINI_API_KEY == "AQ.Ab8RN6L5E2H4-XP6V619IYiWMiaS5yjg1Gb8C4-HRUODeuQn7Q":
    log.warning("لم تقم بوضع مفتاح API صالح، قد يفشل الاتصال بـ Gemini.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Upgraded Models for better accuracy
EMBED_MODEL      = "BAAI/bge-small-en-v1.5" # Stronger embedding
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Reranker
EMBED_DIM        = 384
GEMINI_MODEL     = "gemini-2.5-flash"
FAISS_TOP_K      = 15  # Broad retrieval
FINAL_TOP_K      = 5   # Refined retrieval after reranking

# ================================================================
# 1. Data Structures
# ================================================================
@dataclass
class Document:
    title: str
    category: str
    description: str
    features: str
    detection: str
    mitigation: str
    references: str
    tags: List[str]

    @property
    def embed_text(self) -> str:
        # ما يراه محرك البحث الدلالي (Vector DB)
        parts = [self.title, self.category, self.description, self.features, self.detection]
        return " ".join([p for p in parts if p]).replace("\n", " ")

    @property
    def display_text(self) -> str:
        # ما يتم تمريره إلى Gemini في الـ Prompt
        return (
            f"Title: {self.title}\n"
            f"Category: {self.category}\n"
            f"Description: {self.description}\n"
            f"Features/Indicators: {self.features}\n"
            f"Detection Logic: {self.detection}\n"
            f"Mitigation: {self.mitigation}\n"
            f"References: {self.references}"
        )

# ================================================================
# 2. Initialization (Embedder & Reranker)
# ================================================================
log.info("Loading Embedding Model...")
embedder = SentenceTransformer(EMBED_MODEL)
log.info("Loading Cross-Encoder Reranker...")
reranker = CrossEncoder(RERANK_MODEL)

# ================================================================
# 3. Knowledge Base Loading & Indexing
# ================================================================
def load_knowledge_base(file_path: str) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        log.warning(f"KB file not found: {file_path}. Returning empty.")
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = []
    for item in data:
        docs.append(Document(
            title=item.get("title", "Unknown"),
            category=item.get("category", "General"),
            description=item.get("description", ""),
            features=item.get("features", ""),
            detection=item.get("detection", ""),
            mitigation=item.get("mitigation", ""),
            references=item.get("references", ""),
            tags=item.get("tags", [])
        ))
    log.info(f"Loaded {len(docs)} documents from {file_path}")
    return docs

def build_index(documents: List[Document]) -> tuple[Optional[faiss.Index], List[Document]]:
    if not documents:
        return None, []
    
    texts = [doc.embed_text for doc in documents]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index, documents

# ================================================================
# 4. Advanced Retrieval (FAISS + Reranker + Confidence)
# ================================================================
def retrieve_and_rerank(query: str, index: faiss.Index, documents: List[Document]) -> Tuple[List[Document], float]:
    if index is None or not documents:
        return [], 0.0

    # الخطوة 1: استرجاع واسع باستخدام FAISS
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    k_faiss = min(FAISS_TOP_K, len(documents))
    _, indices = index.search(query_vec, k_faiss)
    
    candidate_docs = [documents[idx] for idx in indices[0] if idx != -1]
    
    if not candidate_docs:
        return [], 0.0

    # الخطوة 2: إعادة الترتيب وحساب الثقة باستخدام Cross-Encoder
    pairs = [[query, doc.embed_text] for doc in candidate_docs]
    scores = reranker.predict(pairs)
    
    # ترتيب الوثائق بناءً على السكور
    scored_docs = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
    
    # حساب نسبة الثقة بناءً على أعلى تطابق
    top_score = scored_docs[0][0]
    confidence_pct = (1 / (1 + np.exp(-top_score))) * 100
    
    # الاحتفاظ بأفضل 5 وثائق فقط
    final_docs = [doc for score, doc in scored_docs[:FINAL_TOP_K] if score > -2.0] 

    return final_docs, confidence_pct

# ================================================================
# 5. Prompt Engineering (Strict Routing & Citation)
# ================================================================
def build_prompt(query: str, context_docs: List[Document]) -> str:
    context_str = ""
    if context_docs:
        blocks = [f"[DOCUMENT {i}]\n{doc.display_text}\n[END DOCUMENT {i}]" for i, doc in enumerate(context_docs, 1)]
        context_str = "\n\n".join(blocks)

    return f"""You are a highly capable Cybersecurity RAG Assistant operating an IDS (Spydera).

Your priority is to answer the user's query using the provided [DOCUMENTS]. 
You are ALLOWED to use your own knowledge ONLY if the documents are insufficient or missing.

════════════════════════════════════════
STRICT ROUTING & SOURCE CITATION RULES
════════════════════════════════════════
You MUST start your response with ONE of the following routing tags on the very first line:

1. <ROUTING: RAG_ONLY> 
   - Use if documents completely answer the question.
   - CITATION RULE: You must cite the documents used like "[Source: Document X - References: Y]".

2. <ROUTING: RAG_MIXED> 
   - Use if you combined documents with your own technical knowledge.
   - CITATION RULE: Cite documents for RAG info. For your added knowledge, you MUST explicitly list standard cybersecurity sources (e.g., OWASP, MITRE, NIST) at the end.

3. <ROUTING: GEMINI_ONLY> 
   - Use if documents were useless/missing.
   - CITATION RULE: You MUST explicitly list credible cybersecurity sources at the bottom under "External Sources Used".

════════════════════════════════════════
RETRIEVED CONTEXT
════════════════════════════════════════
{context_str if context_str else "NO DOCUMENTS RETRIEVED."}

════════════════════════════════════════
USER QUERY
════════════════════════════════════════
{query}
"""

# ================================================================
# 6. Response Generation & Metadata Formatting
# ================================================================
def generate_response(query: str, index: faiss.Index, documents: List[Document]) -> str:
    # 1. جلب الملفات وحساب نسبة الثقة
    retrieved_docs, confidence = retrieve_and_rerank(query, index, documents)
    
    # 2. بناء الـ Prompt وإرساله للـ LLM
    prompt = build_prompt(query, retrieved_docs)
    
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.2, # درجة حرارة منخفضة لضمان الدقة وتجنب الهلوسة
        )
    )
    raw_text = response.text.strip()

    # 3. تحليل الـ Routing Tags لمعرفة مصدر إجابة الـ LLM
    source_type = "UNKNOWN"
    if "<ROUTING: RAG_ONLY>" in raw_text:
        source_type = "RAG_DOCUMENTS (Data Source Only)"
        raw_text = raw_text.replace("<ROUTING: RAG_ONLY>", "").strip()
    elif "<ROUTING: RAG_MIXED>" in raw_text:
        source_type = "RAG + GEMINI KNOWLEDGE (Combined)"
        raw_text = raw_text.replace("<ROUTING: RAG_MIXED>", "").strip()
    elif "<ROUTING: GEMINI_ONLY>" in raw_text:
        source_type = "GEMINI KNOWLEDGE (Context Insufficient)"
        raw_text = raw_text.replace("<ROUTING: GEMINI_ONLY>", "").strip()
        confidence = 0.0 # نسبة ثقة الـ Retrieval تُعتبر صفر لأن النموذج اعتمد على نفسه

    # 4. تنسيق قائمة الملفات اللي استخدمها النظام
    doc_list = "\n".join([f"  {i}- {doc.title}" for i, doc in enumerate(retrieved_docs, 1)])
    if not doc_list:
        doc_list = "  None"

    # 5. إخراج الإجابة بالشكل النهائي المنسق
    final_output = f"""================================================
Source Type:  {source_type}
Confidence:   {confidence:.1f}%
Documents Used:
{doc_list}
================================================

{raw_text}
"""
    return final_output

# ================================================================
# 7. Main Execution (Testing)
# ================================================================
if __name__ == "__main__":
    kb_path = "knowledge_base.json"
    
    # التأكد من وجود ملف قاعدة المعرفة
    if not Path(kb_path).exists():
        log.error(f"Please ensure {kb_path} exists in the same directory.")
    else:
        # تحميل الداتا وبناء الفهرس
        docs = load_knowledge_base(kb_path)
        index, indexed_docs = build_index(docs)
        
        # تجربة النظام بسؤال معقد للـ RAG
        test_query = "Explain how a SYN flood attack can bypass detection systems that rely only on packet rate thresholds, and suggest how a hybrid AI system like Spydera would handle it."
        print(f"\nQuerying: '{test_query}'\n")
        
        # استدعاء دالة الرد وطباعتها
        answer = generate_response(test_query, index, indexed_docs)
        print(answer)
        
        
        """ test_query = "What is Spydera and how does its AI engine detect a DoS UDP Flood attack?"
        print(f"\nQuerying: '{test_query}'\n")
        
        # استدعاء دالة الرد وطباعتها
        answer = generate_response(test_query, index, indexed_docs)
        print(answer)"""