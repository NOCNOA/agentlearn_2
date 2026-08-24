import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ============================================================
# 1. 初始化 LLM Client
# ============================================================

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)

#
MODEL = os.getenv("LLM_MODEL", "DeepSeek-V4-Flash-0731")


# ============================================================
# 2. 假装这是我们的论文数据库
# ============================================================

PAPER_DATABASE = [
    {
        "title": "IGEV-Stereo",
        "keywords": [
            "stereo",
            "cost volume",
            "geometry encoding",
            "iterative"
        ],
    },
    {
        "title": "FoundationStereo",
        "keywords": [
            "stereo",
            "foundation model",
            "zero-shot",
            "cost volume"
        ],
    },
    {
        "title": "RAFT-Stereo",
        "keywords": [
            "stereo",
            "iterative",
            "correlation",
            "GRU"
        ],
    },
    {
        "title": "Large Disparity Stereo Example",
        "keywords": [
            "stereo",
            "large disparity",
            "large disparity range"
        ],
    },
]


# ============================================================
# 3. 真正的 Python Tool
# ============================================================

def search_papers(keyword: str) -> str:
    """
    根据 keyword 搜索论文。
    """

    keyword = keyword.lower()

    results = []

    for paper in PAPER_DATABASE:

        text = (
            paper["title"]
            + " "
            + " ".join(paper["keywords"])
        ).lower()

        if keyword in text:
            results.append(paper)

    if not results:
        return json.dumps(
            {
                "count": 0,
                "papers": [],
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "count": len(results),
            "papers": results,
        },
        ensure_ascii=False,
    )


# ============================================================
# 4. 告诉 LLM：你有哪些工具
# ============================================================

TOOLS = [
    {
        "type": "function",

        "function": {

            "name": "search_papers",

            "description": (
                "Search the local paper database using a keyword. "
                "Use this tool when the user asks about papers, "
                "research methods, or related literature."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "keyword": {
                        "type": "string",
                        "description": (
                            "The keyword used to search papers, "
                            "for example: large disparity"
                        ),
                    }

                },

                "required": ["keyword"],

            },
        },
    }
]


# ============================================================
# 5. Tool Registry
# ============================================================

TOOL_REGISTRY = {
    "search_papers": search_papers,
}


# ============================================================
# 6. Agent
# ============================================================

def run_agent(user_input: str):

    messages = [

        {
            "role": "system",
            "content": (
                "You are a research assistant. "
                "When the user asks about papers or research, "
                "use available tools when necessary. "
                "Do not fabricate search results."
            ),
        },

        {
            "role": "user",
            "content": user_input,
        },

    ]

    max_steps = 5

    for step in range(max_steps):

        print(f"\n========== Agent Step {step + 1} ==========")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # ----------------------------------------------------
        # 情况 1：LLM 不想调用工具
        # ----------------------------------------------------

        if not message.tool_calls:

            print("\nAgent Final Answer:\n")
            print(message.content)

            return message.content

        # ----------------------------------------------------
        # 情况 2：LLM 决定调用工具
        # ----------------------------------------------------

        print("\nAgent wants to use tools:")

        # 必须把 assistant 的 tool call 记录加入历史
        messages.append(
            message.model_dump(exclude_none=True)
        )

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            print(f"Tool: {tool_name}")

            # -----------------------------------------------
            # 解析 LLM 生成的参数
            # -----------------------------------------------

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                tool_result = json.dumps(
                    {
                        "error": "Invalid JSON arguments"
                    }
                )

            else:

                print(
                    "Arguments:",
                    arguments,
                )

                # -------------------------------------------
                # 检查 Tool 是否存在
                # -------------------------------------------

                tool_function = TOOL_REGISTRY.get(
                    tool_name
                )

                if tool_function is None:

                    tool_result = json.dumps(
                        {
                            "error":
                            f"Unknown tool: {tool_name}"
                        }
                    )

                else:

                    # ---------------------------------------
                    # 真正执行 Python Function
                    # ---------------------------------------

                    try:

                        tool_result = tool_function(
                            **arguments
                        )

                    except Exception as e:

                        tool_result = json.dumps(
                            {
                                "error": str(e)
                            }
                        )

            print(
                "Tool Result:",
                tool_result,
            )

            # -----------------------------------------------
            # 把工具结果交回 LLM
            # -----------------------------------------------

            messages.append(
                {
                    "role": "tool",

                    "tool_call_id":
                        tool_call.id,

                    "content":
                        tool_result,
                }
            )

    raise RuntimeError(
        "Agent exceeded maximum number of steps."
    )


# ============================================================
# 7. 程序入口
# ============================================================

if __name__ == "__main__":

    question = (
        "帮我找一下和 large disparity "
        "相关的 stereo matching 论文。"
    )

    run_agent(question)