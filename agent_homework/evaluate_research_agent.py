def evaluate_state(state):
    checks = {
        "has_search_plan": isinstance(
            state.search_plan,
            dict,
        ),
        "has_queries": bool(
            state.search_plan
            and state.search_plan.get("queries")
        ),
        "query_count_within_limit": (
            state.search_plan is not None
            and len(state.search_plan.get("queries", [])) <= 3
        ),
        "has_candidates": bool(
            state.candidate_papers
        ),
        "has_evidence": isinstance(
            state.research_evidence,
            dict,
        ),
        "has_evidence_papers": bool(
            state.research_evidence
            and state.research_evidence.get("papers")
        ),
        "has_final_report": bool(
            state.final_report
            and state.final_report.strip()
        ),
    }

    passed = all(checks.values())

    return {
        "passed": passed,
        "checks": checks,
    }

