import os
import json

from dotenv import load_dotenv
from openai import OpenAI

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.search_arxiv import search_arxiv
from tools.get_arxiv_paper import get_arxiv_paper


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
    },
        {
        "type": "function",
        "function": {
            "name": "get_arxiv_paper",
            "description": (
                "根据 arXiv ID 读取一篇论文的详细信息，"
                "包括完整摘要、作者、发布时间和论文类别。"
                "应先使用 search_arxiv 找到论文 ID，"
                "再用本工具读取最相关论文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": (
                            "论文的 arXiv ID，例如 2608.20788；"
                            "也可以传入 arXiv URL。"
                        ),
                    }
                },
                "required": ["arxiv_id"],
                "additionalProperties": False,
            },
        },
    }
]


# ============================================================
# 5. Tool Registry
# ============================================================

TOOL_REGISTRY = {
    "search_arxiv": search_arxiv,
    "get_arxiv_paper": get_arxiv_paper
}


# ============================================================
# 6. Agent
# ============================================================

def run_agent(user_input: str):

    messages = [

        {
            "role": "system",
            "content": (
                """
                你是一个论文研究助手，可以搜索并阅读 arXiv 论文。

                工作流程：
                1. 首先使用 search_arxiv 搜索相关候选论文。
                2. 根据标题、作者和发布时间判断相关性。
                3. 只对最相关的 1 到 3 篇论文调用 get_arxiv_paper。
                4. 阅读论文摘要后生成最终回答。
                5. 不要重复执行相同查询。
                6. 信息足够后立即停止调用工具。
                7. 不得根据标题猜测论文内容；只有读取详细信息后，
                才能描述论文的方法和贡献。
                """
                
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
        print(f"Messages count: {len(messages)}")
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Current messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
            raise

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
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                }
                for call in message.tool_calls
            ]
        }
        
        # 如果 content 不为空，才添加 content 字段
        if message.content:
            assistant_msg["content"] = message.content
        
        messages.append(assistant_msg)

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

                        result = tool_function(
                            **arguments
                        )
                        
                        # 确保结果是字符串（JSON 格式）
                        if isinstance(result, str):
                            tool_result = result
                        else:
                            tool_result = json.dumps(
                                result,
                                ensure_ascii=False,
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

    question = """
    请搜索与 large-disparity stereo matching 相关的论文，
    选择最相关的两篇，读取摘要后分别说明研究问题、主要方法和可能的局限。
    """

    run_agent(question)