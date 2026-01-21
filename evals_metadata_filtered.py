"""
Metadata-Filtered RAG - Precision Delta Evaluation
Compares retrieval precision with and without metadata filters.

Measures the improvement in precision when using targeted filters
for area-specific, status-specific, or owner-specific questions.

Precision Delta = Filtered Precision - Unfiltered Precision
"""

import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import from our retrieval modules
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


# Test cases with questions that should benefit from filtering
# Each has a question and the filter that should improve precision
PRECISION_DELTA_TEST_CASES = [
    {
        "id": "backend_area",
        "question": "How does authentication and two-factor authentication work in our application?",
        "filter_type": "area",
        "filters": {"area": "Backend"},
        "description": "Backend-specific question with area filter"
    },
    {
        "id": "frontend_area",
        "question": "What is the structure of the frontend application and how does search work?",
        "filter_type": "area",
        "filters": {"area": "Frontend"},
        "description": "Frontend-specific question with area filter"
    },
    {
        "id": "devops_area",
        "question": "How do I access the RDS production database?",
        "filter_type": "area",
        "filters": {"area": "DevOps"},
        "description": "DevOps question with area filter"
    },
    {
        "id": "published_status",
        "question": "What are the finalized and approved engineering practices?",
        "filter_type": "status",
        "filters": {"status": "Published"},
        "description": "Published docs only"
    },
    {
        "id": "chase_norton_owner",
        "question": "What documentation has Chase Norton created about backend features?",
        "filter_type": "owner",
        "filters": {"owner": "Chase Norton"},
        "description": "Chase Norton's documentation"
    },
    {
        "id": "backend_repo",
        "question": "What features and patterns exist in the backend codebase?",
        "filter_type": "repo",
        "filters": {"repo": "liv-app-backend"},
        "description": "Backend repository specific"
    },
    {
        "id": "frontend_repo",
        "question": "What are the frontend application patterns and components?",
        "filter_type": "repo",
        "filters": {"repo": "liv-app-frontend"},
        "description": "Frontend repository specific"
    },
    {
        "id": "inf_ticket",
        "question": "What infrastructure work was done for the RDS migration?",
        "filter_type": "ticket",
        "filters": {"related_ticket": "INF-81"},
        "description": "INF-81 ticket related"
    },
    {
        "id": "combined_backend_published",
        "question": "What are the approved backend engineering practices?",
        "filter_type": "combined",
        "filters": {"area": "Backend", "status": "Published"},
        "description": "Backend + Published combined filter"
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
        filters: dict,
        k: int = DEFAULT_K,
        verbose: bool = False
) -> dict:
    """
    Compare precision with and without filters for a single question.

    Returns:
        Dictionary with precision comparison results
    """
    judge = create_relevance_judge()

    # Retrieve WITHOUT filters (baseline)
    unfiltered_docs = retrieve_with_filter(query=question, top_k=k)
    unfiltered_precision, unfiltered_judgments = calculate_precision(
        question, unfiltered_docs, judge
    )

    # Retrieve WITH filters - extract individual filter params
    filtered_docs = retrieve_with_filter(
        query=question,
        top_k=k,
        area=filters.get("area"),
        status=filters.get("status"),
        owner=filters.get("owner"),
        repo=filters.get("repo"),
        last_edited_by=filters.get("last_edited_by"),
        related_ticket=filters.get("related_ticket")
    )
    filtered_precision, filtered_judgments = calculate_precision(
        question, filtered_docs, judge
    )

    # Calculate delta
    precision_delta = filtered_precision - unfiltered_precision

    result = {
        "question": question,
        "filters": filters,
        "k": k,
        "unfiltered": {
            "precision": unfiltered_precision,
            "relevant_count": sum(unfiltered_judgments),
            "total_docs": len(unfiltered_docs),
            "judgments": unfiltered_judgments
        },
        "filtered": {
            "precision": filtered_precision,
            "relevant_count": sum(filtered_judgments),
            "total_docs": len(filtered_docs),
            "judgments": filtered_judgments
        },
        "precision_delta": precision_delta,
        "improvement": precision_delta > 0
    }

    if verbose:
        result["unfiltered_sources"] = [
            f"{doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('area', 'N/A')})"
            for doc in unfiltered_docs
        ]
        result["filtered_sources"] = [
            f"{doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('area', 'N/A')})"
            for doc in filtered_docs
        ]

    return result


def run_evaluation(
        test_cases: list = None,
        k: int = DEFAULT_K,
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
    print("Metadata-Filtered RAG - Precision Delta Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents): {k}")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)

    results = []
    improvements = 0
    no_change = 0
    regressions = 0

    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case["id"]
        question = test_case["question"]
        filters = test_case["filters"]
        description = test_case["description"]
        filter_type = test_case["filter_type"]

        print(f"\n[{i}/{len(test_cases)}] {test_id}")
        print(f"    Type: {filter_type}")
        print(f"    Q: {question[:55]}...")

        result = evaluate_precision_delta(
            question=question,
            filters=filters,
            k=k,
            verbose=verbose
        )
        result["test_id"] = test_id
        result["description"] = description
        result["filter_type"] = filter_type

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

        print(f"    Unfiltered: {result['unfiltered']['precision']:.0%} ({result['unfiltered']['relevant_count']}/{result['unfiltered']['total_docs']})")
        print(f"    Filtered:   {result['filtered']['precision']:.0%} ({result['filtered']['relevant_count']}/{result['filtered']['total_docs']})")
        print(f"    Delta:      {status}")

    # Calculate aggregate metrics
    avg_unfiltered = sum(r["unfiltered"]["precision"] for r in results) / len(results)
    avg_filtered = sum(r["filtered"]["precision"] for r in results) / len(results)
    avg_delta = sum(r["precision_delta"] for r in results) / len(results)

    # Group by filter type
    by_filter_type = {}
    for r in results:
        ft = r["filter_type"]
        if ft not in by_filter_type:
            by_filter_type[ft] = []
        by_filter_type[ft].append(r["precision_delta"])

    summary = {
        "k": k,
        "num_test_cases": len(test_cases),
        "avg_unfiltered_precision": avg_unfiltered,
        "avg_filtered_precision": avg_filtered,
        "avg_precision_delta": avg_delta,
        "improvements": improvements,
        "no_change": no_change,
        "regressions": regressions,
        "by_filter_type": {
            ft: sum(deltas) / len(deltas)
            for ft, deltas in by_filter_type.items()
        },
        "results": results
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\n📊 Aggregate Metrics:")
    print(f"   Average Unfiltered Precision: {avg_unfiltered:.2%}")
    print(f"   Average Filtered Precision:   {avg_filtered:.2%}")
    print(f"   Average Precision Delta:      {avg_delta:+.2%}")

    print(f"\n📈 Outcomes:")
    print(f"   🟢 Improvements: {improvements}/{len(results)} ({100*improvements/len(results):.0f}%)")
    print(f"   ⚪ No Change:    {no_change}/{len(results)} ({100*no_change/len(results):.0f}%)")
    print(f"   🔴 Regressions:  {regressions}/{len(results)} ({100*regressions/len(results):.0f}%)")

    print(f"\n📋 By Filter Type:")
    for ft, avg in summary["by_filter_type"].items():
        indicator = "🟢" if avg > 0 else "🔴" if avg < 0 else "⚪"
        print(f"   {indicator} {ft}: {avg:+.2%} avg delta")

    print(f"\n📋 Per-Query Breakdown:")
    for r in results:
        delta = r["precision_delta"]
        if delta > 0:
            indicator = "🟢"
        elif delta < 0:
            indicator = "🔴"
        else:
            indicator = "⚪"
        print(f"   {indicator} {r['test_id']}: {r['unfiltered']['precision']:.0%} → {r['filtered']['precision']:.0%} ({delta:+.0%})")

    return summary


def main():
    """Run the precision delta evaluation."""
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    import argparse
    parser = argparse.ArgumentParser(description="Run precision delta evaluation")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include document sources in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    summary = run_evaluation(k=args.k, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
