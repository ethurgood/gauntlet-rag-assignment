"""
Metadata-Filtered RAG - Retrieval Module
Performs metadata filtering BEFORE vector search for more targeted results.

Supported filters:
- area: Filter by area (DevOps, Backend, Frontend, etc.)
- status: Filter by status (Published, In Progress, Needs review)
- owner: Filter by owner name
- owners: Filter by multiple owners
- repo: Filter by repository name
- title: Filter by document title
- last_edited_by: Filter by last editor name
- related_ticket: Filter by related ticket ID
- module: Filter by specific module(s)
- file_type: Filter by file type(s) - .py, .md, .js, etc.
"""

import os
import certifi
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

DEFAULT_TOP_K = 5


def get_mongo_client():
    connection_params = {
        "serverSelectionTimeoutMS": 5000,
        "connectTimeoutMS": 5000,
        "directConnection": True,
        "tls": False,
        "retryWrites": False
    }
    return MongoClient(MONGO_DB_URL, **connection_params)


def get_vector_store():
    """Connect to the MongoDB vector store."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    return vector_store, client


def build_pre_filter(
        area: Optional[str | list[str]] = None,
        status: Optional[str | list[str]] = None,
        owner: Optional[str | list[str]] = None,
        owners: Optional[str | list[str]] = None,
        repo: Optional[str | list[str]] = None,
        title: Optional[str | list[str]] = None,
        last_edited_by: Optional[str | list[str]] = None,
        related_ticket: Optional[str | list[str]] = None,
        module: Optional[str | list[str]] = None,
        file_type: Optional[str | list[str]] = None,
) -> dict:
    """
    Build a MongoDB pre-filter for vector search on engineering docs.
    """
    conditions = []

    def add_condition(field, value):
        if value is not None:
            if isinstance(value, list):
                conditions.append({field: {"$in": value}})
            else:
                conditions.append({field: {"$eq": value}})

    add_condition("area", area)
    add_condition("status", status)
    add_condition("owner", owner)
    add_condition("owners", owners)
    add_condition("repo", repo)
    add_condition("title", title)
    add_condition("last_edited_by", last_edited_by)
    add_condition("related_ticket", related_ticket)
    add_condition("module", module)
    add_condition("file_type", file_type)

    if not conditions:
        return {}
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def retrieve_with_filter(
        query: str,
        top_k: int = DEFAULT_TOP_K,
        area: Optional[str | list[str]] = None,
        status: Optional[str | list[str]] = None,
        owner: Optional[str | list[str]] = None,
        owners: Optional[str | list[str]] = None,
        repo: Optional[str | list[str]] = None,
        title: Optional[str | list[str]] = None,
        last_edited_by: Optional[str | list[str]] = None,
        related_ticket: Optional[str | list[str]] = None,
        module: Optional[str | list[str]] = None,
        file_type: Optional[str | list[str]] = None,
        verbose: bool = False,
) -> list:
    """
    Retrieve documents with metadata pre-filtering for engineering docs.
    """
    vector_store, client = get_vector_store()
    try:
        pre_filter = build_pre_filter(
            area=area,
            status=status,
            owner=owner,
            owners=owners,
            repo=repo,
            title=title,
            last_edited_by=last_edited_by,
            related_ticket=related_ticket,
            module=module,
            file_type=file_type
        )
        if verbose:
            print("Pre-filter:", pre_filter)
        results = vector_store.similarity_search(
            query,
            k=top_k,
            pre_filter=pre_filter
        )
        return results
    finally:
        client.close()

def format_retrieved_context(documents: list) -> str:
    """Format retrieved documents into context string for LLM."""
    context_parts = []

    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get("title", "Unknown")
        area = doc.metadata.get("area", "N/A")
        status = doc.metadata.get("status", "N/A")
        owner = doc.metadata.get("owner", "N/A")
        repo = doc.metadata.get("repo", "N/A")
        chunk_idx = doc.metadata.get("chunk_index", "N/A")

        metadata_str = f"Title: {title} | Area: {area} | Status: {status}"
        if owner:
            metadata_str += f" | Owner: {owner}"
        if repo:
            metadata_str += f" | Repo: {repo}"

        context_parts.append(
            f"[Document {i}]\n"
            f"{metadata_str}\n"
            f"Content:\n{doc.page_content}\n"
        )

    return "\n---\n".join(context_parts)


def debug_collection():
    """Debug function to check MongoDB collection status."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]

    print("=" * 50)
    print("DEBUG: MongoDB Collection Status")
    print("=" * 50)

    doc_count = collection.count_documents({})
    print(f"\nTotal documents in collection: {doc_count}")

    if doc_count > 0:
        sample = collection.find_one()
        print(f"\nSample document fields: {list(sample.keys())}")
        print(f"\nSample metadata:")
        print(f"  title: {sample.get('title', 'N/A')}")
        print(f"  area: {sample.get('area', 'N/A')}")
        print(f"  status: {sample.get('status', 'N/A')}")
        print(f"  owner: {sample.get('owner', 'N/A')}")
        print(f"  owners: {sample.get('owners', 'N/A')}")
        print(f"  repo: {sample.get('repo', 'N/A')}")
        print(f"  last_edited_by: {sample.get('last_edited_by', 'N/A')}")
        print(f"  related_ticket: {sample.get('related_ticket', 'N/A')}")
        print(f"  module: {sample.get('module', 'N/A')}")
        print(f"  file_type: {sample.get('file_type', 'N/A')}")

    client.close()
    return doc_count

def main():
    """Test retrieval with various filters."""
    print("=" * 60)
    print("Metadata-Filtered RAG - Retrieval Test")
    print("=" * 60)

    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Debug collection first
    doc_count = debug_collection()

    if doc_count == 0:
        print("\n❌ No documents found! Run ingestion.py first.")
        return

    # Test queries with different filters
    print("\n" + "=" * 60)
    print("Running Filtered Retrieval Tests")
    print("=" * 60)

    # Test 1: No filter (baseline)
    test_query = "How does authentication and two-factor authentication work in our application?"
    print("\n📝 Test 1: No filter (baseline)")
    print(f"   Query: {test_query}")
    results = retrieve_with_filter(test_query, top_k=3, verbose=True)
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('title', 'Unknown')} | Area: {doc.metadata.get('area', 'N/A')}")

    # Test 2: Filter by area
    print("\n📝 Test 2: Filter by area='Backend'")
    test_query = "How does the backend handle user sessions?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        area="Backend",
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('title', 'Unknown')} | Area: {doc.metadata.get('area', 'N/A')}")

    # Test 3: Filter by status
    print("\n📝 Test 3: Filter by status='Published'")
    test_query = "What are the finalized and approved engineering practices?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        status="Published",
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('title', 'Unknown')} | Status: {doc.metadata.get('status', 'N/A')}")

    # Test 4: Filter by owner
    print("\n📝 Test 4: Filter by owner='Chase Norton'")
    test_query = "What documentation has Chase Norton created about backend features?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        owner="Chase Norton",
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('title', 'Unknown')} | Owner: {doc.metadata.get('owner', 'N/A')}")

    # Test 5: Filter by repo
    print("\n📝 Test 5: Filter by repo='liv-app-backend'")
    test_query = "What features and patterns exist in the backend codebase?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        repo="liv-app-backend",
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('title', 'Unknown')} | Repo: {doc.metadata.get('repo', 'N/A')}")

    # Test 6: Combined filters (area + status)
    print("\n📝 Test 6: Combined filters (area='Backend', status='Published')")
    test_query = "What are the approved backend engineering practices?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        area="Backend",
        status="Published",
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('title', 'Unknown')} | Area: {doc.metadata.get('area', 'N/A')} | Status: {doc.metadata.get('status', 'N/A')}")

if __name__ == "__main__":
    main()