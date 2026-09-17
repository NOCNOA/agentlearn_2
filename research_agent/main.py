import os
import json
import argparse

from dotenv import load_dotenv
from openai import OpenAI

import sys
from contextlib import redirect_stdout
from pathlib import Path

# Windows defaults redirected stdout to the active ANSI code page (commonly
# GBK). arXiv titles and abstracts can contain characters outside GBK, so keep
# both console output and redirected .out files in UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_homework.evaluate_research_agent import evaluate_state
from tools.planner import *

from tools.Tool_desc import TOOLS
from tools.Tool_registry import TOOL_REGISTRY

from dataclasses import dataclass, field

@dataclass
class ResearchState:
    question: str

    search_plan: dict | None = None

    search_results: list[dict] = field(
        default_factory=list
    )

    candidate_papers: list[dict] = field(
        default_factory=list
    )

    research_evidence: dict | None = None

    final_report: str | None = None


load_dotenv()


# ============================================================
# 1. 初始化 LLM Client
# ============================================================

_client: OpenAI | None = None

def validate_string_list(value, field_path):
    if not isinstance(value, list):
        raise ValueError(
            f"{field_path} must be a list."
        )

    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"Every item in {field_path} must be a string."
            )

    return value

def validate_paper_evidence(paper, index):
    path = f"papers[{index}]"

    if not isinstance(paper, dict):
        raise ValueError(f"{path} must be a dictionary.")

    for field_name in ["arxiv_id", "title", "url"]:
        value = paper.get(field_name)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{path}.{field_name} must be a non-empty string."
            )

    source_facts = paper.get("source_facts")

    if not isinstance(source_facts, dict):
        raise ValueError(
            f"{path}.source_facts must be a dictionary."
        )

    for field_name in ["research_problem", "method"]:
        value = source_facts.get(field_name)

        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"{path}.source_facts.{field_name} "
                "must be a string or null."
            )

    validate_string_list(
        source_facts.get("contributions"),
        f"{path}.source_facts.contributions",
    )

    validate_string_list(
        source_facts.get("reported_results"),
        f"{path}.source_facts.reported_results",
    )

    validate_string_list(
        paper.get("explicit_limitations"),
        f"{path}.explicit_limitations",
    )

    validate_string_list(
        paper.get("missing_information"),
        f"{path}.missing_information",
    )

    return paper

def get_client() -> OpenAI:
    """Create the LLM client only when a workflow step needs it."""
    global _client

    if _client is None:
        _client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
        )

    return _client

MODEL = os.getenv("LLM_MODEL", "DeepSeek-V4-Flash-0731")

# ============================================================
# 2. 工具注册
# ============================================================



# ============================================================
# 3. Agent运行主体
# ============================================================
READER_PROMPT = """
Tool-use rules:
- First call get_arxiv_paper for a candidate. Copy its arXiv ID exactly.
- Call get_arxiv_pdf_text only after its abstract was read and only when the
  abstract lacks a technical detail needed for the user's question.
- Read PDF text for at most two papers. Do not retry an identical failed call.

你是论文研究流程中的 Reader Agent。

你的职责：
1. 从候选论文中选择与用户问题最相关的 1 到 3 篇。
2. 必须调用 get_arxiv_paper 读取论文摘要。
3. 只负责选择和读取论文，不负责提取结构化证据或撰写最终报告。
4. 读取足够的论文后立即停止调用工具。
5. 不允许根据标题猜测未读取论文的内容。
6. 如果只有一篇论文高度相关，可以只读取一篇。
7. 不要为了凑够数量而选择弱相关论文。
8. 除非用户要求领域综述，否则优先选择提出具体方法的论文。
9. 完成阅读后只回复 READING_COMPLETE，不要生成分析或 JSON。
"""


EVIDENCE_PROMPT = """
你是论文证据提取器。

你会收到用户问题和若干篇已经读取的论文摘要。
请将摘要整理成一个合法的 JSON 对象。

规则：
1. 只使用传入的论文摘要。
2. source_facts 只能包含摘要直接支持的内容。
3. explicit_limitations 只填写摘要明确说明的局限；没有时返回空列表。
4. missing_information 记录摘要没有提供的信息，不能把缺失信息写成论文局限。
5. 推断必须放在 reasoned_inferences 中，不能混入 source_facts。
6. evidence_summary 中如使用推断，必须使用“可能”“推测”“尚需确认”等措辞。
7. relevance 的理由如果包含任何推断，based_on_inference 必须为 true。
8. cost volume 翻译为“代价体”，不能翻译为“代价卷积”。
9. 只输出一个 JSON 对象，不要输出 Markdown、代码围栏或额外说明。
10. 不得重复输出字段。

输出结构：
{
  "papers": [
    {
      "arxiv_id": "...",
      "title": "...",
      "url": "...",
      "source_facts": {
        "research_problem": null,
        "method": null,
        "contributions": [],
        "reported_results": []
      },
      "explicit_limitations": [],
      "missing_information": [],
      "reasoned_inferences": [
        {
          "claim": "...",
          "basis": "...",
          "confidence": "low"
        }
      ],
      "relevance": {
        "level": "high",
        "reason": "...",
        "based_on_inference": false
      }
    }
  ],
  "evidence_summary": "...",
  "overall_missing_information": []
}
"""


READING_TOOLS = [
    tool
    for tool in TOOLS
    if tool["function"]["name"] in {
        "get_arxiv_paper",
        "get_arxiv_pdf_text",
    }
]


MAX_AGENT_STEPS = 6
MAX_TOOL_CALLS = 6


def extract_research_evidence(
    question: str,
    collected_papers: list[dict],
) -> str:
    if not collected_papers:
        raise RuntimeError(
            "Reader did not collect any papers."
        )

    papers_text = json.dumps(
        collected_papers,
        ensure_ascii=False,
        indent=2,
    )

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": EVIDENCE_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
用户问题：

{question}

已读取的论文摘要：

{papers_text}

请输出结构化研究证据。
""",
            },
        ],
        response_format={
            "type": "json_object",
        },
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Evidence extractor returned empty content."
        )

    return content


def run_reader_agent(
    question: str,
    research_task: str,
):
    messages = [
        {
            "role": "system",
            "content": READER_PROMPT,
        },
        {
            "role": "user",
            "content": research_task,
        },
    ]

    successful_calls = {}
    tool_call_count = 0
    collected_papers = {}

    for step in range(1, MAX_AGENT_STEPS + 1):
        print(f"\n========== Agent Step {step} ==========")
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=READING_TOOLS,
            temperature=0,
        )

        message = response.choices[0].message

        # 没有工具调用，说明模型已经生成最终答案
        if not message.tool_calls:
            print("\nReader finished selecting papers.")
            return extract_research_evidence(
                question=question,
                collected_papers=list(collected_papers.values()),
            )

        print("\nAgent wants to use tools:")

        # 必须把包含 tool_calls 的 assistant 消息加入历史
        messages.append(
            message.model_dump(exclude_none=True)
        )

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments

            print(f"\nTool: {function_name}")
            print(f"Raw Arguments: {raw_arguments}")

            tool_call_count += 1

            # 1. 检查总工具调用预算
            if tool_call_count > MAX_TOOL_CALLS:
                result = {
                    "error": "Tool-call budget exhausted.",
                    "instruction": (
                        "Do not call more tools. "
                        "Answer using existing results."
                    ),
                }

            else:
                # 2. 解析模型生成的 JSON 参数
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    result = {
                        "error": "Invalid tool arguments",
                        "details": str(exc),
                        "raw_arguments": raw_arguments,
                    }

                else:
                    # 创建可比较的工具调用签名
                    call_signature = (
                        function_name,
                        json.dumps(
                            arguments,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )

                    # 3. 成功执行过的调用直接复用结果。
                    # 首次执行失败的调用不会进入缓存，因此仍可重试。
                    if call_signature in successful_calls:
                        result = successful_calls[call_signature]
                        print("Reusing successful tool result.")

                    else:
                        # 4. 查找工具
                        tool_function = TOOL_REGISTRY.get(
                            function_name
                        )

                        if tool_function is None:
                            result = {
                                "error": "Unknown tool",
                                "tool": function_name,
                            }

                        else:
                            # 5. 防止工具异常终止整个 Agent
                            try:
                                result = tool_function(**arguments)

                                if isinstance(result, dict) and "error" not in result:
                                    successful_calls[call_signature] = result

                                    arxiv_id = result.get("arxiv_id")
                                    if (
                                        function_name == "get_arxiv_paper"
                                        and arxiv_id
                                    ):
                                        collected_papers[arxiv_id] = result
                                        print(
                                            "Collected papers: "
                                            f"{len(collected_papers)}"
                                        )

                                    elif (
                                        function_name == "get_arxiv_pdf_text"
                                        and arxiv_id in collected_papers
                                    ):
                                        collected_papers[arxiv_id][
                                            "pdf_excerpt"
                                        ] = result
                                        print(
                                            "Attached PDF text to paper: "
                                            f"{arxiv_id}"
                                        )
                            except Exception as exc:
                                result = {
                                    "error": "Tool execution failed",
                                    "tool": function_name,
                                    "details": str(exc),
                                }

            print("Tool Result:")
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            # 每一个 tool_call 都必须对应一条 tool 消息
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                }
            )

        # 工具预算已经用完，不再进入下一轮工具调用
        if tool_call_count >= MAX_TOOL_CALLS:
            break

    print("\nReader reached its tool or step limit.")
    return extract_research_evidence(
        question=question,
        collected_papers=list(collected_papers.values()),
    )

def parse_json_output(content: str) -> dict:
    content = content.strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise RuntimeError(
            f"No JSON object found:\n{content}"
        )

    json_text = content[start:end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Agent returned invalid JSON:\n{json_text}"
        ) from exc
def validate_research_evidence(evidence: dict) -> dict:
    if not isinstance(evidence, dict):
        raise ValueError(
            "Research evidence must be a dictionary."
        )

    papers = evidence.get("papers")
    evidence_summary = evidence.get("evidence_summary")
    overall_missing_information = evidence.get(
        "overall_missing_information"
    )

    if not isinstance(papers, list):
        raise ValueError(
            "research_evidence.papers must be a list."
        )

    if not papers:
        raise ValueError(
            "research_evidence.papers cannot be empty."
        )

    if not isinstance(evidence_summary, str):
        raise ValueError(
            "research_evidence.evidence_summary must be a string."
        )

    if not isinstance(overall_missing_information, list):
        raise ValueError(
            "overall_missing_information must be a list."
        )

    for item in overall_missing_information:
        if not isinstance(item, str):
            raise ValueError(
                "Every overall_missing_information item "
                "must be a string."
            )
    validated_papers = []

    for index, paper in enumerate(papers):
        validated_papers.append(
            validate_paper_evidence(paper, index)
        )
    return evidence

WRITER_PROMPT = """
你是论文调研工作流中的 Writer。

你只能根据 Reader 提供的结构化证据撰写中文报告。

证据使用规则：

1. source_facts 可以作为论文明确内容陈述。
2. explicit_limitations 只能作为论文明确局限陈述。
3. missing_information 只能描述为“摘要未提供”或“尚无法确认”。
4. 不能把 missing_information 描述成论文缺陷。
5. reasoned_inferences 必须使用“可能、推测、从摘要可推断”等措辞。
6. 不得把 reasoned_inferences 改写成确定事实。
7. 不得提及未被 Reader 调用工具读取的论文。
8. 如果证据不足以回答问题，必须直接说明证据不足。
9. 每篇论文使用传入的 URL。
10. 不要提及 Planner、Reader、JSON 或内部工作流。

报告结构：
- 核心结论
- 高度相关论文
- 方法比较
- 有依据的推断
- 摘要未提供的信息
- 对用户问题的直接回答
"""

def run_writer(
    question: str,
    research_evidence: dict,
) -> str:

    evidence_text = json.dumps(
        research_evidence,
        ensure_ascii=False,
        indent=2,
    )

    writer_task = f"""
用户最初的研究问题：

{question}

Reader 提取的研究证据：

{evidence_text}

请根据这些证据撰写最终研究报告。
"""

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": WRITER_PROMPT,
            },
            {
                "role": "user",
                "content": writer_task,
            },
        ],
        temperature=0,
        # Writer 不需要 tools
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Writer returned empty content."
        )

    return content

def run_research_workflow(
    question: str,
) -> ResearchState:

    state = ResearchState(
        question=question
    )

    # 1. Planner
    state.search_plan = create_search_plan(
        state.question
    )

    print("\n========== Search Plan ==========")
    print(
        json.dumps(
            state.search_plan,
            ensure_ascii=False,
            indent=2,
        )
    )

    # 2. 执行搜索
    state.search_results = execute_search_plan(
        state.search_plan
    )

    # 3. 选择候选论文
    state.candidate_papers = (
        rank_candidate_papers(
            state.search_results,
            limit=10,
        )
    )

    if not state.candidate_papers:
        raise RuntimeError(
            "No candidate papers were found because all arXiv searches "
            "failed or returned no results. Check the Planned Search "
            "errors above and retry later."
        )

    # 4. 构造 Reader 任务
    research_task = build_research_task(
        state.question,
        state.search_plan,
        state.candidate_papers,
    )

    # 5. Reader 选择并读取论文
    reader_content = run_reader_agent(
        question=state.question,
        research_task=research_task,
    )

    # 6. 解析结构化证据
    parsed_evidence = parse_json_output(
        reader_content
    )

    state.research_evidence = validate_research_evidence(
        parsed_evidence
    )

    print("\n========== Research Evidence ==========")
    print(
        json.dumps(
            state.research_evidence,
            ensure_ascii=False,
            indent=2,
        )
    )

    # 7. Writer
    state.final_report = run_writer(
        question=state.question,
        research_evidence=state.research_evidence,
    )

    return state

def run_default_research() -> None:
    question = """
    请调研关于超分辨率的问题。
    """

    state = run_research_workflow(question)
    evaluation = evaluate_state(state)

    print("\n========== Evaluation ==========")
    print(
        json.dumps(
            evaluation,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n========== Final Report ==========")
    print(state.final_report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the complete workflow log to this UTF-8 file.",
    )
    args = parser.parse_args()

    if args.output is None:
        run_default_research()
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        with redirect_stdout(output_file):
            run_default_research()


if __name__ == "__main__":
    main()
