"""
Metadata-Filtered RAG - Generation Module
Uses vector search with metadata filters to generate answers.

Supports filtering by area, status, owner, repo, and other metadata fields.
"""

import os
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import retrieval logic from retrieval_metadata_filtered module
from retrieval_metadata_filtered import retrieve_with_filter, format_retrieved_context

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a technical assistant that answers questions about LIV engineering documents.
Search Method: Vector search with metadata filtering

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information, acknowledge what you can answer and what's missing.

Context:
{context}

Question: {question}

Answer:"""


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
        area: Optional[str] = None,
        status: Optional[str] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        last_edited_by: Optional[str] = None,
        related_ticket: Optional[str] = None,
        verbose: bool = False,
) -> dict:
    """
    Generate an answer using metadata-filtered vector search.

    Args:
        question: The question to answer
        top_k: Number of documents to retrieve
        area: Filter by area (DevOps, Backend, Frontend, etc.)
        status: Filter by status (Published, In Progress, Needs review)
        owner: Filter by owner name
        repo: Filter by repository name
        last_edited_by: Filter by last editor
        related_ticket: Filter by related ticket ID
        verbose: Include retrieved documents in response

    Returns:
        Dictionary with answer, sources, and search method
    """
    # Use retrieval_metadata_filtered's vector_search_with_filters function
    documents = retrieve_with_filter(
        query=question,
        top_k=top_k,
        area=area,
        status=status,
        owner=owner,
        repo=repo,
        last_edited_by=last_edited_by,
        related_ticket=related_ticket
    )

    # Build filter description
    filters_applied = []
    if area:
        filters_applied.append(f"area={area}")
    if status:
        filters_applied.append(f"status={status}")
    if owner:
        filters_applied.append(f"owner={owner}")
    if repo:
        filters_applied.append(f"repo={repo}")
    if last_edited_by:
        filters_applied.append(f"last_edited_by={last_edited_by}")
    if related_ticket:
        filters_applied.append(f"related_ticket={related_ticket}")

    filter_str = f" with filters: {', '.join(filters_applied)}" if filters_applied else ""

    if not documents:
        return {
            "answer": f"I couldn't find any relevant information{filter_str}.",
            "sources": [],
            "search_method": f"Vector search{filter_str}",
            "filters": {
                "area": area,
                "status": status,
                "owner": owner,
                "repo": repo,
                "last_edited_by": last_edited_by,
                "related_ticket": related_ticket,
            }
        }

    context = format_retrieved_context(documents)
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
                "repo": doc.metadata.get("repo", "N/A"),
            }
            for doc in documents
        ],
        "search_method": f"Vector search{filter_str}",
        "filters": {
            "area": area,
            "status": status,
            "owner": owner,
            "repo": repo,
            "last_edited_by": last_edited_by,
            "related_ticket": related_ticket,
        }
    }

    if verbose:
        response["retrieved_documents"] = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]

    return response


def interactive_mode():
    """Run interactive Q&A with metadata filtering."""
    print("\n" + "=" * 60)
    print("Metadata-Filtered RAG - Interactive Q&A")
    print("=" * 60)
    print("\n📌 Using existing documents (no ingestion needed)")
    print(f"   Collection: {DB_NAME}.{COLLECTION_NAME}")
    print("\nCommands:")
    print("  filter:area=Backend       - Set area filter")
    print("  filter:status=Published   - Set status filter")
    print("  filter:owner=John         - Set owner filter")
    print("  filter:repo=backend-api   - Set repo filter")
    print("  clear                     - Clear all filters")
    print("  quit                      - Exit")
    print("-" * 60)

    # Current filters
    filters = {
        "area": None,
        "status": None,
        "owner": None,
        "repo": None,
        "last_edited_by": None,
        "related_ticket": None,
    }

    while True:
        print()
        active_filters = [f"{k}={v}" for k, v in filters.items() if v]
        if active_filters:
            print(f"🔍 Active filters: {', '.join(active_filters)}")
        else:
            print("🔍 No filters active")

        user_input = input(">> ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if user_input.lower() == 'clear':
            filters = {k: None for k in filters}
            print("✅ All filters cleared")
            continue

        if user_input.startswith('filter:'):
            try:
                filter_part = user_input.split(':', 1)[1]
                key, value = filter_part.split('=', 1)
                key = key.strip()
                value = value.strip()

                if key in filters:
                    filters[key] = value
                    print(f"✅ Filter set: {key}={value}")
                else:
                    print(f"❌ Unknown filter: {key}")
                    print(f"   Available: {', '.join(filters.keys())}")
            except Exception as e:
                print(f"❌ Invalid format. Use 'filter:key=value'")
            continue

        print("\n🔍 Searching with metadata filters...")
        print("🤖 Generating answer...\n")

        try:
            result = generate_answer(
                user_input,
                **filters
            )

            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])

            print(f"\n📚 Sources ({len(result['sources'])} documents):")
            for s in result["sources"]:
                print(f"  • {s['title']} ({s['area']}) - {s['status']}")

            print(f"\n🔍 {result['search_method']}")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run example queries or interactive mode."""
    print("=" * 60)
    print("Metadata-Filtered RAG - Generation")
    print("=" * 60)
    print(f"   Reusing collection: {DB_NAME}.{COLLECTION_NAME}")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    examples = [
        {
            "question": "How do I access the production database?",
            "filters": {"area": "DevOps"},
            "note": "Filtered to DevOps documents only"
        },
        {
            "question": "What are the authentication best practices?",
            "filters": {"area": "Backend", "status": "Published"},
            "note": "Filtered to Published Backend documents"
        },
        {
            "question": "How does the search component work?",
            "filters": {"area": "Frontend", "repo": "liv-app-frontend"},
            "note": "Filtered to Frontend repo documents"
        },
    ]

    print("\n📋 Example Queries:")
    for i, ex in enumerate(examples, 1):
        filter_str = ", ".join([f"{k}={v}" for k, v in ex['filters'].items()])
        print(f"  {i}. {ex['question']}")
        print(f"     Filters: {filter_str}")
        print(f"     Note: {ex['note']}")
        print()

    print("-" * 50)
    choice = input("Enter 1-3 for examples, 'i' for interactive, or your question: ").strip()

    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3']:
        ex = examples[int(choice) - 1]
        print(f"\n📝 Question: {ex['question']}")
        print(f"💡 Note: {ex['note']}")
        print(f"🔍 Filters: {ex['filters']}")

        print("\n🔍 Retrieving with metadata filters...")
        print("🤖 Generating answer...\n")

        result = generate_answer(ex['question'], **ex['filters'])

        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])

        print(f"\n📚 Sources:")
        for s in result["sources"]:
            print(f"  • {s['title']} ({s['area']}) - {s['status']}")
            if s.get('owner'):
                print(f"    Owner: {s['owner']}")
            if s.get('repo'):
                print(f"    Repo: {s['repo']}")

        print(f"\n🔍 {result['search_method']}")
    elif choice:
        print(f"\n📝 Question: {choice}")
        print("\n🔍 Retrieving with vector search (no filters)...")

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
