import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)
from agent_homework.minimal_agent import (
    run_agent as run_calculator_agent,
)
from agent_homework.note_agent import (
    run_agent as run_note_agent,
)



import json
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
MODEL = os.getenv(
    "LLM_MODEL",
    "DeepSeek-V4-Flash-0731",
)
_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client

    if _client is None:
        _client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
        )

    return _client
MAX_SUPERVISOR_STEPS = 4
MAX_SUPERVISOR_TOOL_CALLS = 4
SUPERVISOR_PROMPT = """
你是一个任务委托 Supervisor。

你的职责是理解用户问题，选择合适的 Worker，
并根据 Worker 返回的结果回答用户。

规则：
1. 算术或数值计算问题，委托 delegate_to_calculator。
2. 关于 Agent、messages、ResearchState 或笔记内容的问题，
   委托 delegate_to_note_worker。
3. 不要自行完成 Worker 应做的底层任务。
4. Worker 返回结果后，直接基于结果回答用户。
5. 简单问题通常只需委托一个 Worker。
6. 如果无需 Worker，也可以直接回答。
7. 如果用户问题包含多个相互独立的子任务，
   必须委托对应的多个 Worker，再整合结果。
8. 不要因为某个子任务简单，就跳过本应委托的 Worker。
9. 如果后一个子任务依赖前一个 Worker 的结果，
   必须先等待前一个 Worker 返回，再发起后续委托。
10. 不得自行猜测或计算 Worker 尚未返回的数据。
"""
def run_supervisor_agent(user_input: str):
    messages = [
        {
            "role": "system", 
            "content": SUPERVISOR_PROMPT
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]
    tool_call_count = 0
    for step in range(MAX_SUPERVISOR_STEPS):
        print(f"\n==========Supervisor Agent Step {step} ==========")
        # 1. 调用模型
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=SUPERVISOR_TOOLS,
            temperature=0,
        )
        
        # 2. 判断有没有 tool_calls
        message = response.choices[0].message
        print("====model output====")
        print(message.content)
        
        # 3. 没有则返回最终回答
        if not message.tool_calls:
            print(f"\nFinish. Using Tools for {tool_call_count} times")
            return message.content
        
        # 4. 有则保存 assistant 消息
        messages.append(message.model_dump(exclude_none=True))#上一轮信息保存为字典对象，加到messages当中
        for tool_call in message.tool_calls:
            tool_call_count += 1
        
            if tool_call_count > MAX_SUPERVISOR_TOOL_CALLS:
                result = {
                    "error": "Tool-call budget exhausted",
                    "instruction": (
                        "Do not call more tools. "
                        "Answer using existing results."
                    ),
                }
            else:    
                function_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                print(f"\nTool: {function_name}")
                print(f"Raw Arguments: {raw_arguments}")
                
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    result = {
                        "error": "Invalid tool arguments",
                        "details": str(exc),
                        "raw_arguments": raw_arguments,
                    }

                else:
                    tool_function = SUPERVISOR_TOOL_REGISTRY.get(
                        function_name
                    )
                    if tool_function is None:
                        result = {
                            "error": "Unknown tool",
                            "tool": function_name,
                        }
                    else:
                        try:
                            result = tool_function(**arguments)
                        except Exception as exc:
                            result = {
                                "error": "Tool execution failed",
                                "tool": function_name,
                                "details": str(exc),
                            }
            print(f"Tool Result: {result}")
            # 5. 将 tool result 写回 messages
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


    raise RuntimeError("Agent reached its step limit")




def delegate_to_calculator(task: str) -> dict:
    answer = run_calculator_agent(task)

    return {
        "worker": "calculator",
        "answer": answer,
    }


def delegate_to_note_worker(task: str) -> dict:
    state = run_note_agent(user_input=task)

    return {
        "worker": "note",
        "answer": state["answer"],
    }

SUPERVISOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_calculator",
            "description": (
                "Delegate arithmetic tasks to the calculator worker. "
                "Use it for addition, multiplication, division, "
                "or questions requiring numerical calculation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "A complete and self-contained calculation task "
                            "for the calculator worker."
                        ),
                    }
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_note_worker",
            "description": (
                "Delegate note-search and note-reading questions "
                "to the note worker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "A complete and self-contained question "
                            "about the stored notes."
                        ),
                    }
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
]
SUPERVISOR_TOOL_REGISTRY = {
    "delegate_to_calculator": delegate_to_calculator,
    "delegate_to_note_worker": delegate_to_note_worker,
} 

# def main():
#     note_result = delegate_to_note_worker(
#         "ResearchState 是什么？"
#     )
#     print(note_result)

#     calculator_result = delegate_to_calculator(
#         "请计算 12 乘以 8。"
#     )
#     print(calculator_result)
def main():
    result = run_supervisor_agent(
    "查找工具调用预算示例是多少，再计算其两倍。"
)

    print("\n========== Final Answer ==========")
    print(result)

if __name__ == "__main__":
    main()