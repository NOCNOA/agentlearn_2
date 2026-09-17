import sys
from pathlib import Path
import os,json
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timezone
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from agent_homework.memory_store import (
    forget,
    recall,
    remember,
    search_memories,
)


def remember_memory(key: str, value: str) -> dict:
    remember(key, value)

    return {
        "ok": True,
        "key": key,
        "value": value,
    }


def recall_memory(key: str) -> dict:
    record = recall(key)

    if record is None:
        return {
            "ok": False,
            "key": key,
            "value": None,
        }

    return {
        "ok": True,
        "key": key,
        "value": record["value"],
        "source": record["source"],
        "updated_at": record["updated_at"],
    }
def search_memory(keyword: str) -> dict:
    results = search_memories(keyword)

    return {
        "ok": True,
        "keyword": keyword,
        "count": len(results),
        "memories": results,
    }

def forget_memory(key: str) -> dict:
    deleted = forget(key)

    return {
        "ok": True,
        "key": key,
        "deleted": deleted,
    }
    
MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember_memory",
            "description": (
                "Save an explicit long-term user preference or fact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                    },
                    "value": {
                        "type": "string",
                    },
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Read a saved long-term memory by key."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                    },
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": (
                "Delete a saved long-term memory by key."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                    },
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": (
            "Search saved long-term memories by a keyword. "
            "Use it when the exact memory key is unknown."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                },
            },
            "required": ["keyword"],
            "additionalProperties": False,
        },
    },
},
]

MEMORY_TOOL_REGISTRY = {
    "remember_memory": remember_memory,
    "recall_memory": recall_memory,
    "forget_memory": forget_memory,
    "search_memory": search_memory,
}



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

MEMORY_PROMPT = """
你是一个带长期记忆的助手。

规则：
1. 用户明确说“记住”某项偏好或事实时，
   调用 remember_memory。
2. 用户询问之前保存的信息时，
   必须先调用 recall_memory，不能猜测。
3. 用户明确说“忘记”某项信息时，
   调用 forget_memory。
4. 不要自动保存模型推测、临时闲聊或敏感信息。
5. 工具返回后，根据结果用中文回答用户。
6. 用户明确更新已有偏好或事实时，
   使用相同 key 写入新值，最新明确值覆盖旧值。
7. 用户表达模糊、临时或可能与旧记忆冲突的信息时，
   先请求澄清，不要自动覆盖或合并。
8. 不要把“偶尔、可能、尝试、也许”等描述
   当作稳定的长期偏好保存。
9. 用户询问过去的偏好或事实、但未明确给出记忆 key 时，
   先调用 search_memory。
10. 只能根据 search_memory 或 recall_memory 返回的内容回答，
    不得凭空回忆。
"""
MAX_AGENT_STEPS = 5
MAX_TOOL_CALLS = 4

def run_memory_agent(user_input: str):
    messages = [
        {   
            "role": "system", 
            "content": MEMORY_PROMPT
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]
    tool_call_count = 0
    for step in range(MAX_AGENT_STEPS):
        print(f"\n==========Memory Agent Step {step} ==========")
        # 1. 调用模型
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=MEMORY_TOOLS,
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
        
            if tool_call_count > MAX_TOOL_CALLS:
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
                    tool_function = MEMORY_TOOL_REGISTRY.get(function_name)
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



def main():
    result = run_memory_agent(
        "请回忆我之前的研究方向、"
        "论文年份和输出语言偏好。"
    )

    print(result)

if __name__ == "__main__":
    main()
    # run_memory_agent("我偶尔也会看 image segment 论文。")