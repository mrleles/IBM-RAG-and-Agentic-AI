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

# Query Fusion Retriever