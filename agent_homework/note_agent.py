from dotenv import load_dotenv
from openai import OpenAI
import os, json
load_dotenv()
MODEL = os.getenv("LLM_MODEL", "DeepSeek-V4-Flash-0731")

NOTES = {
    "1": "Agent 通过工具调用访问外部能力。",
    "2": "messages 保存模型当前对话历史。",
    "3": "ResearchState 保存整个工作流状态。",
}

def search_notes(keyword):
    related_note_ids = []

    for note_id, content in NOTES.items():
        if keyword in content:
            related_note_ids.append(note_id)

    return {
        "ok": True,
        "keyword": keyword,
        "note_ids": related_note_ids,
        "count": len(related_note_ids),
    }

def get_note(note_id):
    if note_id not in NOTES:
        return {
            "ok": False,
            "error": "Note not found",
            "note_id": note_id,
        }

    return {
        "ok": True,
        "note_id": note_id,
        "content": NOTES[note_id],
    }

TOOLS = [
    {
        "type": "function",
        "function":{
            "name":"search_notes",
            "description":"return the list of related notes' ids",
            "parameters":{
                "type": "object",
                "properties":{
                    "keyword":{
                        "type":"string",
                        "description":"keywords of your query"
                    }
                },
                "required": ["keyword"],
            }
            },
    },
    {
        "type": "function",
        "function":{
            "name":"get_note",
            "description":"return the related notes' content of by notes'ids",
            "parameters":{
                "type": "object",
                "properties":{
                    "note_id":{
                        "type":"string",
                        "description":"id of note"
                    }
                },
                "required": ["note_id"],
            }
            },
    }
]
TOOL_REGISTRY = {
    "search_notes":search_notes,
    "get_note":get_note
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

READER_PROMPT="""你需要根据笔记回答一个简答的问题，对于问题你需要做一下简单的关键词提取
                你可以尝试调用工具来解决。
                可能要分为若干步计算，你需要调用若干次工具。
                确定完成后结束即可"""
   
   
MAX_AGENT_STEPS = 10
MAX_TOOL_CALLS = 40        
MAX_TOOL_RETRIES = 1
#一个封装好的模型调用部分
def call_model(state):
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=state["messages"],
        tools=TOOLS,
        temperature=0,
    )

    message = response.choices[0].message
    state["last_message"] = message

    state["trace"].append(
        {
            "step": state["step"],
            "node": "call_model",
            "has_tool_calls": bool(message.tool_calls),
        }
    )

    return state


def execute_tools(state, message,i):
    for tool_call in message.tool_calls:
        state["tool_call_count"] += 1

        if state["tool_call_count"] > MAX_TOOL_CALLS:
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
                tool_function = TOOL_REGISTRY.get(function_name)

                if tool_function is None:
                    result = {
                        "error": "Unknown tool",
                        "tool": function_name,
                    }
                else:
                    for retry_index in range(MAX_TOOL_RETRIES + 1):
                        try:
                            result = tool_function(**arguments)
                            break
                        except Exception as exc:
                            result = {
                                "ok": False,
                                "error": "Tool execution failed",
                                "tool": function_name,
                                "details": str(exc),
                                "attempts": retry_index + 1,
                            }
                    print(f"Tool Result: {result}")
                    state["trace"].append(
                        {
                            "step": state["step"],
                            "node": "execute_tools",
                            "tool_call_times": state["tool_call_count"],
                            "tool": function_name,
                            "arguments": arguments,
                            "result": result,
                        }
                    )
        # 5. 将 tool result 写回 messages
        state["messages"].append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                ),
            }
        )
    return state

def route_after_model(state):
    message = state["last_message"]

    if message.tool_calls:
        return "tools"

    return "end"

def run_agent(user_input: str):

    state = {
        "messages": [
            {"role": "system", "content": READER_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "step": 0,
        "trace": [],
        "tool_call_count": 0,
        "answer": None,
        "last_message": None,
    }
    for i in range(MAX_AGENT_STEPS):
        print(f"\n========== Agent Step {i} ==========")
        #1\模型调用
        state["step"] = i + 1
        state = call_model(state)
        message = state["last_message"]
        print("====model output====")
        print(message.content)
        # 3. 没有则返回最终回答
        next = route_after_model(state)
        state["trace"].append(
            {
                "step": state["step"],
                "node": "route_after_model",
                "route": next,
            }
        )
        if next == "end":
            print( f"\nFinish. Using Tools for {state['tool_call_count']} times" )
            state["answer"] = message.content
            return state
        # 4. 有则保存 assistant 消息
        if next == "tools":
            state["messages"].append(message.model_dump(exclude_none=True))#上一轮信息保存为字典对象，加到messages当中
            state = execute_tools(state, message, i)

    raise RuntimeError("Agent reached its step limit")
    
def main():
    question = "请读取编号为 999 的笔记，并告诉我内容。"
    state = run_agent(question)

    print("\n========== Trace ==========")
    for item in state["trace"]:
        print(json.dumps(item, ensure_ascii=False, indent=2))

    print("\n========== Final Answer ==========")
    print(state["answer"])


if __name__ == "__main__":
    main()