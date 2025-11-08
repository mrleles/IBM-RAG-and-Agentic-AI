'''
!pip install llama-index==0.12.49 \
    llama-index-embeddings-huggingface==0.5.5 \
    llama-index-llms-ibm==0.4.0 \
    llama-index-retrievers-bm25==0.5.2 \
    sentence-transformers==5.0.0 \
    rank-bm25==0.2.2 \
    PyStemmer==2.2.0.3 \
    ibm-watsonx-ai==1.3.31 | tail -n 1
'''
import os
import json
from typing import List, Optional
import asyncio
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    AutoMergingRetriever,
    RecursiveRetriever,
    QueryFusionRetriever
)
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Advanced retriever imports
from llama_index.retrievers.bm25 import BM25Retriever

# IBM WatsonX LlamaIndex integration
from ibm_watsonx_ai import APIClient
from llama_index.llms.ibm import WatsonxLLM

# Sentence transformers
from sentence_transformers import SentenceTransformer

# Statistical libraries for fusion techniques
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - some advanced fusion features will be limited")

print("✅ All imports successful!")

def create_watson_llm():
    try:
        api_client = APIClient({'url': "url"})
        llm = WatsonxLLM(
            model_id="model",
            url="url",
            project_id="project",
            api_client=api_client,
            temperature=0.9
        )
        return llm
    except Exception as e:
        print(f"Error: {e}")

        from llama_index.core.llms.mock import MockLLM
        return MockLLM(max_tokens=512)
    

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

llm = create_watson_llm()

Settings.llm = llm
Settings.embed_model = embed_model

# Sample data for the lab - AI/ML focused documents
SAMPLE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds hidden patterns in data without labeled examples.",
    "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
    "Generative AI can create new content including text, images, code, and more.",
    "Large language models are trained on vast amounts of text data to understand and generate human-like text."
]

# Consistent query examples used throughout the lab
DEMO_QUERIES = {
    "basic": "What is machine learning?",
    "technical": "neural networks deep learning", 
    "learning_types": "different types of learning",
    "advanced": "How do neural networks work in deep learning?",
    "applications": "What are the applications of AI?",
    "comprehensive": "What are the main approaches to machine learning?",
    "specific": "supervised learning techniques"
}

print(f"📄 Loaded {len(SAMPLE_DOCUMENTS)} sample documents")
print(f"🔍 Prepared {len(DEMO_QUERIES)} consistent demo queries")
for i, doc in enumerate(SAMPLE_DOCUMENTS[:3], 1):
    print(f"{i}. {doc}")
print("...")

class AdvancedRetrieversLab:
    def __init__(self):
        self.documents = [Document(text=text) for text in SAMPLE_DOCUMENTS]
        self.nodes = SentenceSplitter().get_nodes_from_documents(self.documents)

        self.vector_index = VectorIndexRetriever.from_documents(self.documents)
        self.document_summary_index = DocumentSummaryIndex.from_documents(self.documents)
        self.keyword_index = KeywordTableIndex.from_documents(self.documents)

lab = AdvancedRetrieversLab()

# Vector Index Retriever
vector_retriever = VectorIndexRetriever(
    index=lab.vector_index,
    similarity_top_k=3
)

alt_retriever = lab.vector_index.as_retriever(similarity_top_k=3)

query = DEMO_QUERIES["basic"]
nodes = vector_retriever.retrieve(query)

for i, node in enumerate(nodes, 1):
    print(f"{i}. Score: {nodes.score:.4f}")
    print(f" Text: {node.text[:100]}")

# BM25
try:
    import Stemmer

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=lab.nodes,
        similarity_top_k=3,
        stemmer=Stemmer.Stemmer("english"),
        language="english"
    )

    query = DEMO_QUERIES["technical"]
    nodes = bm25_retriever.retrieve(query)

    for i, node in enumerate(nodes, 1):
        score = node.score if hasattr(node, 'score') and node.score else 0
        print(f"{i}. BM25 Score: {score:.4f}")
        print(f" Text: {node.text[:100]}...")

        text_lower = node.text.lower()
        query_terms = query.lower().split()
        found_terms = [term for term in query_terms if term in text_lower]
        if found_terms:
            print(f" Found terms: {found_terms}")

except ImportError:
    print("BM25Retriever requires 'pip install PyStemmer'")

# Hybrid Retriver with Vector Similarity with BM25
vector_retriever = lab.vector_index.as_retriever(similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=lab.nodes, similarity_top_k=10
)

def hybrid_retrieve(query, top_k=5):
    vector_results = vector_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)

    vector_scores = {}
    bm25_scores = {}
    all_nodes = {}

    max_vector_score = max([r.score for r in vector_results]) if vector_results else 1
    for result in vector_results:
        text_key = result.text.strip()
        normalized_score = result.score / max_vector_score
        vector_scores[text_key] = normalized_score
        all_nodes[text_key] = result

    max_bm25_score = max([r.score for r in bm25_results]) if bm25_results else 1
    for result in bm25_results:
        text_key = result.text.strip()
        normalized_score = result.score /max_bm25_score
        bm25_scores[text_key] = normalized_score
        all_nodes[text_key] = result

    hybrid_results = []
    for text_key in all_nodes:
        vector_score = vector_scores.get(text_key, 0)
        bm25_score = bm25_scores.get(text_key, 0)
        hybrid_score = 0.7 * vector_score + 0.3 * bm25_score

        hybrid_results.append({
            'node': all_nodes[text_key],
            'vector_score': vector_score,
            'bm25_score': bm25_score,
            'hybrid_score': hybrid_score
        })

    hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return hybrid_results[:top_k]

test_queries = [
    "What is machine learning?",
    "neural networks deep learning", 
    "supervised learning techniques"
]

for query in test_queries:
    results = hybrid_retrieve(query, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. Hybrid Score: {result['hybrid_score']:.3f}")
        print(f" Vector: {result['vector_score']:.3f}, BM25: {result['bm25_score']:.3f}")
        print(f" Text: {result['node'].text[:80]}...")

# Document Summary Index Retriever
query = DEMO_QUERIES["learning_types"]

doc_summary_retriever_llm = DocumentSummaryIndexLLMRetriever(
    lab.document_summary_index, choice_top_k=3
)

try:
    nodes_llm = doc_summary_retriever_llm.retrieve(query)
    for i, node in enumerate(nodes_llm[:2], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Document summary)")
        print(f" Text: {node.text[:80]}...")
except Exception as e:
    print(f" Error: {str(e)[:100]}...")

doc_summary_retriever_embedding = DocumentSummaryIndexEmbeddingRetriever(
    lab.document_summary_index, similarity_top_k=3
)

try:
    nodes_emb = doc_summary_retriever_embedding.retrieve(query)
    for i, node in enu(nodes_emb[:2], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score' and node.score else f"{i}. (Document summary)"))
        print(f" Text: {node.text[:80]}...")
except Exception as e:
    print(f"Error: {str(e)[:100]}...")

# Auto merging retriever
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512,256,128]
)

hier_nodes = node_parser.get_nodes_from_documents(lab.documents)

from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore

docstore = SimpleDocumentStore()
docstore.add_documents(hier_nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

base_index = VectorStoreIndex(hier_nodes, storage_context=storage_context)
base_retriever = base_index.as_retriever(similarity_top_k=6)

auto_merging_retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True
)

query = DEMO_QUERIES["advanced"]
nodes = auto_merging_retriever.retrieve(query)

for i, node in enumerate(nodes[:3], 1):
    print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Auto-merged)")
    print(f" Text: {node.text[:120]}...")

# Recursive Retriever
docs_with_refs = []
for i, doc in enumerate(lab.documents):
    ref_doc = Document(
        text=doc.text,
        metadata={
            "doc_id": f"doc_{i}",
            "references": [f"doc_{j}" for j in range(len(lab.documents)) if j != i][:2]
        }
    )
    docs_with_refs.append(ref_doc)

ref_index = VectorStoreIndex.from_documents(docs_with_refs)

retriever_dict = {
    f"doc_{i}": ref_index.as_retriever(similarity_top_k=1)
    for i in range(len(docs_with_refs))
}

base_retriever = ref_index.as_retriever(similarity_top_k=2)

retriever_dict["vector"] = base_retriever

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict=retriever_dict,
    query_engine_dict={},
    verbose=True
)

query = DEMO_QUERIES["applications"]
try:
    nodes = recursive_retriever.retrieve(query)
    for i, node in enumerate(nodes[:3], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Recursive)")
        print(f" Text: {node.text[:100]}...")
except Exception as e:
    print(f"Query: {query}")
    print(f"Recursive retriever demo: {str(e)}")
    print("Note: Recursive retriever requires specific node reference setup")

# Query Fusion Retriever - Reciprocal Rank Fusion - RRF
base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)
query = DEMO_QUERIES["comprehensive"]
try:
    rrf_query_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=True
    )

    nodes = rrf_query_fusion.retrieve(query)

    for i, node in enumerate(nodes, 1):
        print(f"{i}. Final RRF Score: {node.score:.4f}")
        print(f" Text: {node.text[:100]}...")

except Exception as e:
    print(f"Error: {e}")

# Query Fusion Retriever - Relative Score
base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)
query = DEMO_QUERIES["comprehensive"]

try:
    rel_score_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="relative_score",
        use_async=False,
        verbose=False
    )
    nodes = rel_score_fusion.retrieve(query)

    for i, node in enumerate(nodes, 1):
        print(f"{i}. Combined Relative Score: {node.score:.4f}")
        print(f" Text: {node.text[:100]}...")

except Exception as e:
    print(f"Error: {e}")

# Query Fusion Retriever - Distribution Based
base_retriever = lab.vector_index.as_retriever(similarity_top_k=8)
query = DEMO_QUERIES["comprehensive"]

try:
    dis_fusion = QueryFusionRetriever(
        [base_index],
        similarity_top_k=3,
        num_queries=3,
        mode="dist_based_score",
        use_async=False,
        verbose=False
    )
    nodes = dis_fusion.retrieve(query)

    for i, node in enumerate(nodes, 1):
        print(f"{i}. Statistically Normalized Score: {node.score:.4f}")
        print(f" Text: {node.text[:100]}...")

except Exception as e:
    print(f"Error: {e}")

# Production RAG Pipeline
class ProductionRAGPipeline:
    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        self.vector_retriever = index.as_retriever(similarity_top_k=5)

    def _route_query(self, question):
        if any(word in question.lower() for word in ["what", "explain", "describe"]):
            return "semantic"
        elif any(word in question.lower() for word in ["list", "types", "examples"]):
            return "comprehensive"
        else:
            return "semantic"
        
    def query(self, question, strategy="auto"):
        try:
            if strategy == "auto":
                strategy = self._route_query(question)
            if strategy == "semantic":
                retriever = self.vector_retriever
                top_k = 3
            elif strategy == "comprehensive":
                retriever = self.vector_retriever
                top_k = 5
            else:
                retriever = self.vector_retriever
                top_k = 3

            relevant_docs = retriever.retrieve(question)
            context = "\n\n".join([doc.text for doc in relevant_docs[:top_k]])
            prompt = f"""Based on the following context, please answer the question:
Context:
{context}

Question: {question}

Answer:"""
        
            try:
                response = self.llm.complete(prompt)
                return {
                    "answer": response.text,
                    "strategy": strategy,
                    "num_docs": len(relevant_docs),
                    "status": "success"
                }
            except Exception as e:
                return {
                    "answer": f"Based on the retrieved documents: {context[:200]}...",
                    "strategy": strategy,
                    "num_docs": len(relevant_docs),
                    "status": f"llm_error: {str(e)}"
                }
        except Exception as e:
            return {
                "answer": "I encountered an error processing your question.",
                "strategy": strategy,
                "num_docs": 0,
                "status": f"error: {str(e)}"
            }
        
    def evalueate(self, test_queries):
        results = []
        for query in test_queries:
            result = self.query(query)
            results.append({
                "query": query,
                "result": result,
                "success": result["status"] == "success"
            })
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        return {
            "success_rate": success_rate,
            "results": results
        }
    
pipeline = ProductionRAGPipeline(lab.vector_index, llm)

test_queries = [
    "What is machine learning?",
    "List different types of learning algorithms",
    "Explain neural networks"
]

for query in test_queries:
    result = pipeline.query(query)
    print(f"\nQuery: {query}")
    print(f"Strategy: {result['strategy']}")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer'][:100]}...")

evaluation = pipeline.evalueate(test_queries)
print(f"\nPipeline Success Rate: {evaluation['success_rate']:.2%}")