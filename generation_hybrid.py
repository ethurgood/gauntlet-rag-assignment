"""
Hybrid Search RAG - Generation Module
Uses hybrid retrieval (BM25 + Vector) to generate answers.

"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import retrieval logic from retrieval_hybrid module
from retrieval_hybrid import hybrid_search, load_documents_for_bm25

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration (reusing existing collection)
DB_NAME = "rag_assignment"
COLLECTION_NAME = "engineering_docs_rag"  # or "naive_rag"
INDEX_NAME = "metadata_filtered_index"  # or "naive"

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# Default hybrid weights
DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_VECTOR_WEIGHT = 0.5

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a technical assisant that answers questions about LIV engineering documents.
Search Method: Hybrid (combining keyword and semantic search)

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information, acknowledge what you can answer and what's missing.

Context:
{context}

Question: {question}

Answer:"""

# Global cache for BM25 documents (expensive to rebuild)
_bm25_documents_cache = None


def format_context(documents: list) -> str:
    """Format documents into context string."""
    parts = []
    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get("title", "Unknown")
        area = doc.metadata.get("area", "N/A")
        status = doc.metadata.get("status", "N/A")
        owner = doc.metadata.get("owner", "N/A")
        repo = doc.metadata.get("repo", "N/A")

        metadata_str = f"Title: {title} | Area: {area} | Status: {status}"
        if owner:
            metadata_str += f" | Owner: {owner}"
        if repo:
            metadata_str += f" | Repo: {repo}"

        parts.append(
            f"[Document {i}]\n"
            f"{metadata_str}\n"
            f"Content:\n{doc.page_content}\n"
        )
    return "\n---\n".join(parts)


def create_rag_chain():
    """Create the RAG chain."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

    return prompt | llm | StrOutputParser()


def generate_answer(
        question: str,
        top_k: int = TOP_K,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        verbose: bool = False,
) -> dict:
    """Generate an answer using hybrid RAG."""
    global _bm25_documents_cache

    # Load and cache documents for BM25
    if _bm25_documents_cache is None:
        _bm25_documents_cache = load_documents_for_bm25()

    # Use retrieval_hybrid's hybrid_search function
    documents = hybrid_search(
        query=question,
        k=top_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        documents=_bm25_documents_cache
    )

    if not documents:
        return {
            "answer": "I couldn't find any relevant information.",
            "sources": [],
            "search_method": f"Hybrid (BM25: {bm25_weight}, Vector: {vector_weight})"
        }

    context = format_context(documents)
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    response = {
        "answer": answer,
        "sources": [
            {
                "title": doc.metadata.get("title", "Unknown"),
                "area": doc.metadata.get("area", "N/A"),
                "status": doc.metadata.get("status", "N/A"),
                "owner": doc.metadata.get("owner", "N/A"),
            }
            for doc in documents
        ],
        "search_method": f"Hybrid (BM25: {bm25_weight}, Vector: {vector_weight})"
    }

    if verbose:
        response["retrieved_documents"] = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]

    return response


def interactive_mode():
    """Run interactive Q&A with hybrid search."""
    global _bm25_documents_cache

    print("\n" + "=" * 60)
    print("Hybrid Search RAG - Interactive Q&A")
    print("=" * 60)
    print("\n📌 Using existing documents (no ingestion needed)")
    print(f"   Collection: {DB_NAME}.{COLLECTION_NAME}")
    print("\nCommands:")
    print("  weights:0.7,0.3    - Set BM25/Vector weights")
    print("  quit               - Exit")
    print("-" * 60)

    print("\n🔄 Loading documents for BM25 index...")
    if _bm25_documents_cache is None:
        _bm25_documents_cache = load_documents_for_bm25()
    print(f"   Loaded {len(_bm25_documents_cache)} documents")

    bm25_weight = DEFAULT_BM25_WEIGHT
    vector_weight = DEFAULT_VECTOR_WEIGHT

    while True:
        print()
        print(f"⚖️  Current weights: BM25={bm25_weight}, Vector={vector_weight}")
        user_input = input(">> ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if user_input.startswith('weights:'):
            try:
                weights = user_input.split(':')[1].split(',')
                bm25_weight = float(weights[0])
                vector_weight = float(weights[1])
                print(f"✅ Weights updated: BM25={bm25_weight}, Vector={vector_weight}")
            except Exception:
                print("❌ Invalid format. Use 'weights:0.7,0.3'")
            continue

        print("\n🔀 Searching with hybrid retrieval...")
        print("🤖 Generating answer...\n")

        try:
            result = generate_answer(
                user_input,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )

            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])

            print(f"\n📚 Sources ({len(result['sources'])} documents):")
            for s in result["sources"]:
                print(f"  • {s['title']} ({s['area']}) - {s['status']}")

            print(f"\n🔀 {result['search_method']}")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run example queries or interactive mode."""
    print("=" * 60)
    print("Hybrid Search RAG - Generation")
    print("=" * 60)
    print(f"   Reusing collection: {DB_NAME}.{COLLECTION_NAME}")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    examples = [
        {
            "question": "How do I access the RDS production database?",
            "note": "BM25 helps with exact terms like 'RDS' and 'database'"
        },
        {
            "question": "What are the best practices for implementing authentication in the backend?",
            "note": "Vector helps with semantic meaning and concepts"
        },
        {
            "question": "How do I connect to the production database using SSM and a bastion host?",
            "note": "Hybrid combines specific terms (SSM, 3bastion, database) with semantic context"
        },
    ]

    print("\n📋 Example Queries:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex['question'][:45]}... ({ex['note']})")

    print("\n" + "-" * 50)
    choice = input("Enter 1-3 for examples, 'i' for interactive, or your question: ").strip()

    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3']:
        ex = examples[int(choice) - 1]
        print(f"\n📝 Question: {ex['question']}")
        print(f"💡 Note: {ex['note']}")

        print("\n🔀 Retrieving with hybrid search...")
        print("🤖 Generating answer...\n")

        result = generate_answer(ex['question'])

        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])

        print(f"\n📚 Sources:")
        for s in result["sources"]:
            print(f"  • {s['title']} ({s['area']}) - {s['status']}")
    elif choice:
        print(f"\n📝 Question: {choice}")
        print("\n🔀 Retrieving with hybrid search...")

        result = generate_answer(choice)

        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])

        print(f"\n📚 Sources:")
        for s in result["sources"]:
            print(f"  • {s['title']} ({s['area']}) - {s['status']}")
    else:
        print("\n👋 No input. Run again to try!")


if __name__ == "__main__":
    main()
