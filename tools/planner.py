import json
import os, re, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.search_arxiv import search_arxiv
MODEL = os.getenv("LLM_MODEL", "DeepSeek-V4-Flash-0731")
PLANNER_PROMPT = """
你是论文调研规划器。

你的任务不是回答用户问题，也不是搜索论文，而是生成一个简短的
arXiv 搜索计划。

要求：
1. research_goal 用中文描述调研目标。
2. 生成最多 3 个英文搜索词。
3. 不要生成含义高度重复的搜索词。
4. query 应适合用于 arXiv 论文搜索。
5. purpose 用中文解释该搜索词的作用。
6. 只输出 JSON，不要输出 Markdown 或其他文字。

输出格式：
{
  "research_goal": "...",
  "queries": [
    {
      "query": "...",
      "purpose": "..."
    }
  ]
}
"""
def validate_search_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        raise ValueError("Search plan must be a dictionary.")

    research_goal = plan.get("research_goal")
    queries = plan.get("queries")

    if not isinstance(research_goal, str):
        raise ValueError("research_goal must be a string.")

    if not isinstance(queries, list):
        raise ValueError("queries must be a list.")

    validated_queries = []
    seen_queries = set()

    for item in queries[:3]:
        if not isinstance(item, dict):
            continue

        query = item.get("query")
        purpose = item.get("purpose")

        if not isinstance(query, str) or not query.strip():
            continue

        if not isinstance(purpose, str):
            purpose = ""

        normalized_query = query.strip().lower()

        if normalized_query in seen_queries:
            continue

        seen_queries.add(normalized_query)

        validated_queries.append(
            {
                "query": query.strip(),
                "purpose": purpose.strip(),
            }
        )

    if not validated_queries:
        raise ValueError(
            "Planner did not generate any valid queries."
        )

    return {
        "research_goal": research_goal.strip(),
        "queries": validated_queries,
    }

def normalize_arxiv_id(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip()

    if "/abs/" in arxiv_id:
        arxiv_id = arxiv_id.split("/abs/")[-1]

    if "/pdf/" in arxiv_id:
        arxiv_id = arxiv_id.split("/pdf/")[-1]

    arxiv_id = arxiv_id.removesuffix(".pdf")

    # 删除版本号：2608.12345v2 → 2608.12345
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    return arxiv_id

def execute_search_plan(
    search_plan: dict,
    max_results_per_query: int = 5,
) -> list[dict]:

    all_papers = {}

    for index, query_item in enumerate(
        search_plan["queries"],
        start=1,
    ):
        query = query_item["query"]
        purpose = query_item["purpose"]

        print(
            f"\n========== Planned Search {index} =========="
        )
        print(f"Query: {query}")
        print(f"Purpose: {purpose}")

        try:
            result = search_arxiv(
                query=query,
                max_results=max_results_per_query,
            )
        except Exception as exc:
            print(f"Search failed: {exc}")
            continue

        # 如果你的 search_arxiv 返回的是 JSON 字符串，
        # 需要先转换成 Python 字典
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                print("Search returned invalid JSON.")
                continue

        papers = result.get("papers", [])

        print(f"Found: {len(papers)} papers")

        for paper in papers:
            arxiv_id = paper.get("arxiv_id")

            # 兼容旧版本搜索工具
            if not arxiv_id:
                url = paper.get("url", "")
                arxiv_id = normalize_arxiv_id(url)

            if not arxiv_id:
                continue

            normalized_id = normalize_arxiv_id(arxiv_id)

            if normalized_id not in all_papers:
                paper_copy = dict(paper)

                paper_copy["arxiv_id"] = normalized_id
                paper_copy["matched_queries"] = [query]
                paper_copy["search_purposes"] = [purpose]

                all_papers[normalized_id] = paper_copy

            else:
                existing = all_papers[normalized_id]

                if query not in existing["matched_queries"]:
                    existing["matched_queries"].append(query)

                if purpose not in existing["search_purposes"]:
                    existing["search_purposes"].append(purpose)

    merged_papers = list(all_papers.values())

    print("\n========== Search Summary ==========")
    print(f"Unique papers: {len(merged_papers)}")

    return merged_papers

def create_search_plan(question: str) -> dict:
    client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": PLANNER_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Planner returned empty content.")

    # 有些模型可能仍然添加 Markdown 代码块
    content = content.strip()

    if content.startswith("```json"):
        content = content.removeprefix("```json")
        content = content.removesuffix("```")
        content = content.strip()

    elif content.startswith("```"):
        content = content.removeprefix("```")
        content = content.removesuffix("```")
        content = content.strip()

    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Planner returned invalid JSON:\n{content}"
        ) from exc

    return validate_search_plan(plan)

def rank_candidate_papers(
    papers: list[dict],
    limit: int = 15,
) -> list[dict]:

    ranked = sorted(
        papers,
        key=lambda paper: len(
            paper.get("matched_queries", [])
        ),
        reverse=True,
    )

    return ranked[:limit]

if __name__ == "__main__":

    question = """
    请调研适合处理 large disparity 的 stereo matching 方法，
    重点关注它们如何减少大范围视差搜索的计算量。
    """

    plan = create_search_plan(question)

    print(
        json.dumps(
            plan,
            ensure_ascii=False,
            indent=2,
        )
    )
    
def build_research_task(
    question: str,
    search_plan: dict,
    candidate_papers: list[dict],
) -> str:

    plan_text = json.dumps(
        search_plan,
        ensure_ascii=False,
        indent=2,
    )

    candidates_text = json.dumps(
        candidate_papers,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
            用户的研究问题：

            {question}

            搜索计划：

            {plan_text}

            Python 已经执行搜索并得到以下候选论文：

            {candidates_text}

            请完成以下任务：

            1. 根据标题和命中的查询选择最相关的 1 到 3 篇论文。
            2. 对选中的论文调用 get_arxiv_paper。
            3. 必须读取详细摘要后才能描述论文方法。
            4. 比较这些论文与用户问题的关系。
            5. 如果候选论文不够相关，请明确说明。
            6. 信息足够后停止调用工具并给出最终回答。
            """