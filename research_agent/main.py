import os
import json

from dotenv import load_dotenv
from openai import OpenAI

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.search_arxiv import search_arxiv


load_dotenv()


# ============================================================
# 1. 初始化 LLM Client
# ============================================================

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)

MODEL = os.getenv("LLM_MODEL", "DeepSeek-V4-Flash-0731")

# ============================================================
# 2. 告诉 LLM：你有哪些工具
# ============================================================

TOOLS = [
    {
        "type": "function",

        "function": {

            "name": "search_arxiv",

            "description": (
                "Search the online paper research tool using a keyword. "
                "Use this tool when the user asks about papers, "
                "research methods, or related literature."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description": (
                            "The keyword used to search papers, "
                            "for example: large disparity"
                        ),
                    },
                    
                    "max_results": {
                        "type": "int",
                        "description": (
                            "The number that papers return, "
                            "for example: 5"
                        ),
                    },
                },

                "required": ["query", "max_results"],

            },
        },
    }
]


# ============================================================
# 5. Tool Registry
# ============================================================

TOOL_REGISTRY = {
    "search_arxiv": search_arxiv,
}


# ============================================================
# 6. Agent
# ============================================================

def run_agent(user_input: str):

    messages = [

        {
            "role": "system",
            "content": (
                "你是论文搜索助手。"

                "最多调用 搜索工具 1 次。"
                "得到足够结果后必须停止搜索，并根据已有结果回答。"
                "如果搜索结果不足，应明确说明，而不是无限更换关键词。"
                "不要重复使用相同或高度相似的查询。"
                
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
        "帮我找一下最多5篇和 large disparity "
        "相关的 stereo matching 论文。"
    )

    run_agent(question)