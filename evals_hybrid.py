"""
Hybrid Search RAG - Precision Delta Evaluation
Compares retrieval precision with and without keyword searching (BM25).

Measures the improvement in precision when adding keyword search (BM25)
to vector search for queries that benefit from exact term matching.

Precision Delta = Hybrid Precision - Vector-Only Precision
"""

import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import from our retrieval modules
from retrieval_hybrid import hybrid_search, load_documents_for_bm25
from retrieval_metadata_filtered import retrieve_with_filter

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Evaluation configuration
DEFAULT_K = 5
JUDGE_MODEL = "gpt-4o-mini"

# LLM-as-Judge prompt for relevance assessment
RELEVANCE_JUDGE_PROMPT = """You are a relevance judge. Determine if the retrieved document is relevant to answering the given question.

A document is RELEVANT if it contains information that would help answer the question.
A document is NOT RELEVANT if it contains no useful information for the question.

Question: {question}

Retrieved Document:
{document}

Is this document relevant? Respond with ONLY "RELEVANT" or "NOT_RELEVANT"."""


# Test cases with questions that should benefit from keyword matching
# These queries contain specific technical terms, acronyms, or exact phrases
PRECISION_DELTA_TEST_CASES = [
    {
        "id": "rds_database_access",
        "question": "How do I access the RDS production database?",
        "description": "Specific technical term (RDS) that benefits from exact matching",
        "expected_benefit": "high"
    },
    {
        "id": "ssm_bastion",
        "question": "What is the process for SSM bastion access?",
        "description": "Technical acronyms (SSM) and specific terms",
        "expected_benefit": "high"
    },
    {
        "id": "two_factor_authentication",
        "question": "authentication flow two factor",
        "description": "Keyword query with specific terms",
        "expected_benefit": "high"
    },
    {
        "id": "feature_flags_implementation",
        "question": "feature flags configuration setup",
        "description": "Specific technical terms as keywords",
        "expected_benefit": "medium"
    },
    {
        "id": "backend_auth_flow",
        "question": "How does the backend authentication system work?",
        "description": "Technical terms with semantic context",
        "expected_benefit": "medium"
    },
    {
        "id": "frontend_search_component",
        "question": "Where is the search functionality implemented in the frontend?",
        "description": "Specific component query",
        "expected_benefit": "medium"
    },
    {
        "id": "inf_ticket_migration",
        "question": "INF-81 database migration RDS",
        "description": "Ticket number and specific terms - pure keyword query",
        "expected_benefit": "high"
    },
    {
        "id": "conceptual_architecture",
        "question": "What is the overall system architecture and design philosophy?",
        "description": "Conceptual/semantic query - may not benefit from keywords",
        "expected_benefit": "low"
    },
    {
        "id": "best_practices_general",
        "question": "What are the engineering best practices and guidelines?",
        "description": "General semantic query",
        "expected_benefit": "low"
    },
    {
        "id": "exact_error_message",
        "question": "Connection refused port 5432 PostgreSQL",
        "description": "Exact error message with specific terms",
        "expected_benefit": "high"
    },
]


def create_relevance_judge():
    """Create the LLM judge for assessing relevance."""
    prompt = ChatPromptTemplate.from_template(RELEVANCE_JUDGE_PROMPT)

    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def judge_relevance(judge_chain, question: str, document_content: str) -> bool:
    """Use LLM to judge if a document is relevant to the question."""
    response = judge_chain.invoke({
        "question": question,
        "document": document_content
    })

    return "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper()


def calculate_precision(question: str, documents: list, judge) -> tuple[float, list[bool]]:
    """
    Calculate precision for a set of retrieved documents.

    Returns:
        Tuple of (precision_score, list_of_relevance_judgments)
    """
    if not documents:
        return 0.0, []

    judgments = []
    for doc in documents:
        is_relevant = judge_relevance(judge, question, doc.page_content)
        judgments.append(is_relevant)

    precision = sum(judgments) / len(judgments)
    return precision, judgments


def evaluate_precision_delta(
        question: str,
        k: int = DEFAULT_K,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        documents: list = None,
        verbose: bool = False
) -> dict:
    """
    Compare precision with and without keyword search (BM25) for a single question.

    Returns:
        Dictionary with precision comparison results
    """
    judge = create_relevance_judge()

    # Load documents once for BM25 (caching)
    if documents is None:
        documents = load_documents_for_bm25()

    # Retrieve WITHOUT BM25 (vector-only baseline)
    vector_docs = retrieve_with_filter(query=question, top_k=k)
    vector_precision, vector_judgments = calculate_precision(
        question, vector_docs, judge
    )

    # Retrieve WITH BM25 (hybrid search)
    hybrid_docs = hybrid_search(
        query=question,
        k=k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        documents=documents
    )
    hybrid_precision, hybrid_judgments = calculate_precision(
        question, hybrid_docs, judge
    )

    # Calculate delta
    precision_delta = hybrid_precision - vector_precision

    result = {
        "question": question,
        "k": k,
        "bm25_weight": bm25_weight,
        "vector_weight": vector_weight,
        "vector_only": {
            "precision": vector_precision,
            "relevant_count": sum(vector_judgments),
            "total_docs": len(vector_docs),
            "judgments": vector_judgments
        },
        "hybrid": {
            "precision": hybrid_precision,
            "relevant_count": sum(hybrid_judgments),
            "total_docs": len(hybrid_docs),
            "judgments": hybrid_judgments
        },
        "precision_delta": precision_delta,
        "improvement": precision_delta > 0
    }

    if verbose:
        result["vector_sources"] = [
            f"{doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('area', 'N/A')})"
            for doc in vector_docs
        ]
        result["hybrid_sources"] = [
            f"{doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('area', 'N/A')})"
            for doc in hybrid_docs
        ]

    return result


def run_evaluation(
        test_cases: list = None,
        k: int = DEFAULT_K,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        verbose: bool = False
) -> dict:
    """
    Run precision delta evaluation on all test cases.

    Returns:
        Dictionary with aggregate results
    """
    if test_cases is None:
        test_cases = PRECISION_DELTA_TEST_CASES

    print("=" * 60)
    print("Hybrid Search RAG - Precision Delta Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents): {k}")
    print(f"  - BM25 weight: {bm25_weight}")
    print(f"  - Vector weight: {vector_weight}")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")

    # Load documents once for all evaluations
    print(f"\n📚 Loading documents for BM25...")
    documents = load_documents_for_bm25()
    print(f"   Loaded {len(documents)} documents")

    print("\n" + "-" * 60)

    results = []
    improvements = 0
    no_change = 0
    regressions = 0

    # Group by expected benefit
    by_benefit = {"high": [], "medium": [], "low": []}

    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case["id"]
        question = test_case["question"]
        description = test_case["description"]
        expected_benefit = test_case.get("expected_benefit", "medium")

        print(f"\n[{i}/{len(test_cases)}] {test_id}")
        print(f"    Expected benefit: {expected_benefit}")
        print(f"    Q: {question[:55]}...")

        result = evaluate_precision_delta(
            question=question,
            k=k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            documents=documents,
            verbose=verbose
        )
        result["test_id"] = test_id
        result["description"] = description
        result["expected_benefit"] = expected_benefit

        results.append(result)

        # Track outcomes
        delta = result["precision_delta"]
        if delta > 0:
            improvements += 1
            status = f"🟢 +{delta:.0%}"
        elif delta < 0:
            regressions += 1
            status = f"🔴 {delta:.0%}"
        else:
            no_change += 1
            status = "⚪ 0%"

        print(f"    Vector-only: {result['vector_only']['precision']:.0%} ({result['vector_only']['relevant_count']}/{result['vector_only']['total_docs']})")
        print(f"    Hybrid:      {result['hybrid']['precision']:.0%} ({result['hybrid']['relevant_count']}/{result['hybrid']['total_docs']})")
        print(f"    Delta:       {status}")

        # Group by expected benefit
        by_benefit[expected_benefit].append(delta)

    # Calculate aggregate metrics
    avg_vector = sum(r["vector_only"]["precision"] for r in results) / len(results)
    avg_hybrid = sum(r["hybrid"]["precision"] for r in results) / len(results)
    avg_delta = sum(r["precision_delta"] for r in results) / len(results)

    summary = {
        "k": k,
        "bm25_weight": bm25_weight,
        "vector_weight": vector_weight,
        "num_test_cases": len(test_cases),
        "avg_vector_precision": avg_vector,
        "avg_hybrid_precision": avg_hybrid,
        "avg_precision_delta": avg_delta,
        "improvements": improvements,
        "no_change": no_change,
        "regressions": regressions,
        "by_expected_benefit": {
            benefit: sum(deltas) / len(deltas) if deltas else 0.0
            for benefit, deltas in by_benefit.items()
        },
        "results": results
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\n📊 Aggregate Metrics:")
    print(f"   Average Vector-Only Precision: {avg_vector:.2%}")
    print(f"   Average Hybrid Precision:      {avg_hybrid:.2%}")
    print(f"   Average Precision Delta:       {avg_delta:+.2%}")

    print(f"\n📈 Outcomes:")
    print(f"   🟢 Improvements: {improvements}/{len(results)} ({100*improvements/len(results):.0f}%)")
    print(f"   ⚪ No Change:    {no_change}/{len(results)} ({100*no_change/len(results):.0f}%)")
    print(f"   🔴 Regressions:  {regressions}/{len(results)} ({100*regressions/len(results):.0f}%)")

    print(f"\n📋 By Expected Benefit:")
    for benefit, avg in summary["by_expected_benefit"].items():
        indicator = "🟢" if avg > 0 else "🔴" if avg < 0 else "⚪"
        print(f"   {indicator} {benefit}: {avg:+.2%} avg delta")

    print(f"\n📋 Per-Query Breakdown:")
    for r in results:
        delta = r["precision_delta"]
        if delta > 0:
            indicator = "🟢"
        elif delta < 0:
            indicator = "🔴"
        else:
            indicator = "⚪"
        expected = r["expected_benefit"]
        print(f"   {indicator} {r['test_id']} ({expected}): {r['vector_only']['precision']:.0%} → {r['hybrid']['precision']:.0%} ({delta:+.0%})")

    return summary


def main():
    """Run the precision delta evaluation."""
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    import argparse
    parser = argparse.ArgumentParser(description="Run hybrid search precision delta evaluation")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("--bm25-weight", type=float, default=0.5, help="Weight for BM25 results (default: 0.5)")
    parser.add_argument("--vector-weight", type=float, default=0.5, help="Weight for vector results (default: 0.5)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include document sources in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    summary = run_evaluation(
        k=args.k,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
        verbose=args.verbose
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
