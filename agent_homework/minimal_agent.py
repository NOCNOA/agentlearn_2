from dotenv import load_dotenv
from openai import OpenAI
import os, json
load_dotenv()
MODEL = os.getenv("LLM_MODEL", "DeepSeek-V4-Flash-0731")


def add_numbers(a: int, b: int) -> int:
    return a + b

def multi_numbers(a: int, b: int) -> int:
    return a * b

TOOLS = [
    {
        "type": "function",
        "function":{
            "name": "add_numbers",
            "description":"calculate the sum of a and b.",
            "parameters":{
                "type": "object",
                "properties": {
                    "a":{
                        "type": "integer",
                        "description": "first parameter",
                    },
                    "b":{
                        "type": "integer",
                        "description": "second parameter",
                    },   
                },
                "required": ["a", "b"],
            },

        },
    },
    {
        "type": "function",
        "function":{
            "name": "multi_numbers",
            "description":"calculate the multi of a and b.",
            "parameters":{
                "type": "object",
                "properties": {
                    "a":{
                        "type": "integer",
                        "description": "first parameter",
                    },
                    "b":{
                        "type": "integer",
                        "description": "second parameter",
                    },   
                },
            "required": ["a", "b"],
            },

        }
    }
]

TOOL_REGISTRY = {
    "add_numbers": add_numbers,
    "multi_numbers": multi_numbers,
}



_client: OpenAI | None = None
def get_client() -> OpenAI:
    """Create the LLM client only when a workflow step needs it."""
    global _client

    if _client is None:
        _client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
        )

    return _client

READER_PROMPT="""你需要回答一个简答的计算问题，你可以尝试调用工具来解决。
                计算可能要分为若干步计算，你需要调用若干次工具。
                确定计算完成后结束即可"""
def run_agent(user_input: str):
    messages = [
        {
            "role": "system",
            "content": READER_PROMPT,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    for step in range(5):
        print(f"\n========== Agent Step {step} ==========")
        # 1. 调用模型
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        # 2. 判断有没有 tool_calls
        message = response.choices[0].message
        
        # 3. 没有则返回最终回答
        if not message.tool_calls:
            print("\nFinish.")
            return message.content
        
        # 4. 有则保存 assistant 消息
        messages.append(message.model_dump(exclude_none=True))#上一轮信息保存为字典对象，加到messages当中
        for tool_call in message.tool_calls:
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
                tool_function = TOOL_REGISTRY.get(function_name)
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
    question = "先计算 27 加 58，再把结果乘以 3"
    result = run_agent(question)

    print("\n========== Final Answer ==========")
    print(result)


if __name__ == "__main__":
    main()