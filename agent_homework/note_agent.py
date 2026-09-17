from dotenv import load_dotenv
from openai import OpenAI
import os, json
load_dotenv()
MODEL = os.getenv("LLM_MODEL", "DeepSeek-V4-Flash-0731")

NOTES = {
    "1": "Agent 通过工具调用访问外部能力。",
    "2": "messages 保存模型当前对话历史。",
    "3": "ResearchState 保存整个工作流状态。",
    "4": "当前练习中的工具调用预算示例为 4 次。",
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
def save_checkpoint(state, file_path):
    checkpoint = {
        "messages": state["messages"],
        "trace": state["trace"],
        "tool_call_count": state["tool_call_count"],
        "answer": state["answer"],
        "step": state["step"],
        "status": state["status"],
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            checkpoint,
            file,
            ensure_ascii=False,
            indent=2,
        )
def load_checkpoint(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    # last_message 是运行时临时状态，不从 JSON 中恢复
    checkpoint["last_message"] = None

    return checkpoint
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

def run_agent(
    user_input=None,state=None,checkpoint_path=None,pause_after_tools=False,):
    if state is None:
        if user_input is None:
            raise ValueError(
                "user_input is required when no state is provided"
            )

        state = {
            "messages": [
                {"role": "system", "content": READER_PROMPT},
                {"role": "user", "content": user_input},
            ],
            "trace": [],
            "tool_call_count": 0,
            "answer": None,
            "last_message": None,
            "step": 0,
            "status": "running",
        }

    if state["status"] == "completed":
        return state
    for i in range(state["step"], MAX_AGENT_STEPS):
            state["step"] = i + 1
            print(f"\n========== Agent Step {i} ==========")
            #1\模型调用
            state = call_model(state)
            message = state["last_message"]
            print("====model output====")
            print(message.content)
            # 3. 没有则返回最终回答
            next_node = route_after_model(state)
            state["trace"].append(
                {
                    "step": state["step"],
                    "node": "route_after_model",
                    "route": next_node,
                }
            )
            if next_node == "end":
                print( f"\nFinish. Using Tools for {state['tool_call_count']} times" )
                state["answer"] = message.content
                state["status"] = "completed"
                if checkpoint_path is not None:
                    save_checkpoint(state, checkpoint_path)
                return state
            # 4. 有则保存 assistant 消息
            if next_node == "tools":
                state["messages"].append(message.model_dump(exclude_none=True))#上一轮信息保存为字典对象，加到messages当中
                state = execute_tools(state, message, i)
                if checkpoint_path is not None:
                    save_checkpoint(state, checkpoint_path)

                if pause_after_tools:
                    print("\nWorkflow paused after tools.")
                    return state
    raise RuntimeError("Agent reached its step limit")

def main1():
    state = load_checkpoint(
        "agent_homework/note_agent_checkpoint.json"
    )

    state = run_agent(state=state)

    print(state["status"])
    print(state["answer"])

def main():
    question = "请读取编号为 1 的笔记，并告诉我内容。"
    state = run_agent(question)
    save_checkpoint(
        state,
        "agent_homework/note_agent_checkpoint.json",
    )
    print("\n========== Trace ==========")
    for item in state["trace"]:
        print(json.dumps(item, ensure_ascii=False, indent=2))

    print("\n========== Final Answer ==========")
    print(state["answer"])

def main2():
    question = "请找到所有与 Agent 状态有关的笔记，读取相关内容并总结。"

    state = run_agent(
        user_input=question,
        checkpoint_path="agent_homework/note_agent_checkpoint.json",
        pause_after_tools=True,
    )

    print("status:", state["status"])
    print("step:", state["step"])
    print("answer:", state["answer"])

def main3():
    question = "请找到所有与 Agent 状态有关的笔记，读取相关内容并总结。"

    state = run_agent(
        user_input=question,
        checkpoint_path="agent_homework/note_agent_checkpoint.json",
        pause_after_tools=True,
    )

    print("status:", state["status"])
    print("step:", state["step"])
    print("answer:", state["answer"])
    
def main4():
    path = "agent_homework/note_agent_checkpoint.json"
    state = load_checkpoint(
        path
    )

    state = run_agent(state=state,checkpoint_path=path)

    print("status:", state["status"])
    print("step:", state["step"])
    print("answer:", state["answer"])
if __name__ == "__main__":
    main4()