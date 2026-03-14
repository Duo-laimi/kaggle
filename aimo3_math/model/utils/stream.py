from openai.types.chat import ChatCompletionMessage
import json
from .format import extract_think, exclude_think, exclude_tool_call, extract_tool_call

# api stream
# content reasoning_content tool_calls

def collect_api_stream(stream, verbose=True):
    content = []
    reasoning_content = []
    tool_calls = {}
    is_answering = False
    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
    for chunk in stream:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            if verbose:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                print(delta.content, end="", flush=True)
            content.append(delta.content)
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            if not is_answering and verbose:
                print(delta.reasoning_content, end="", flush=True)
            reasoning_content.append(delta.reasoning_content)
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tool_call_chunk in delta.tool_calls:
                call_index = tool_call_chunk.index
                tool_call_chunk.function.arguments = tool_call_chunk.function.arguments or ""
                if call_index not in tool_calls:
                    tool_calls[call_index] = tool_call_chunk
                else:
                    tool_calls[call_index].function.arguments += tool_call_chunk.function.arguments
    content = "".join(content).strip()
    if len(content) == 0:
        content = None
    reasoning_content = "".join(reasoning_content).strip()
    if len(reasoning_content) == 0:
        reasoning_content = None
    if len(tool_calls) == 0:
        tool_calls = None
    else:
        tool_calls = [tool_call.to_dict() for tool_call in tool_calls.values()]
    return content, reasoning_content, tool_calls

def collect_model_stream(stream, verbose=True):
    reasoning_parts = []
    accumulated_text = ""
    
    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
    
    for chunk in stream:
        accumulated_text += chunk
    
    accumulated_text = accumulated_text.strip()
    
    if "<think>" in accumulated_text and "</think>" in accumulated_text:
        start = accumulated_text.find("<think>") + len("<think>")
        end = accumulated_text.find("</think>")
        reasoning_content = accumulated_text[start:end]
        if verbose:
            print(reasoning_content, end="", flush=True)
        reasoning_parts.append(reasoning_content)
    
    if verbose:
        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
    
    content = exclude_tool_call(accumulated_text)
    content = exclude_think(content)
    reasoning_content = extract_think(accumulated_text)
    
    tool_call_str = extract_tool_call(accumulated_text)
    if tool_call_str is not None and len(tool_call_str) > 0:
        try:
            tool_calls = [json.loads(tool_call_str)]
        except json.JSONDecodeError:
            tool_calls = None
    else:
        tool_calls = None
    
    if content is None or len(content) == 0:
        content = None
    if reasoning_content is None or len(reasoning_content) == 0:
        reasoning_content = None
        
    if verbose and content:
        print(content, end="", flush=True)
    
    return content, reasoning_content, tool_calls



def construct_completion_message(content, reasoning_content, tool_calls):
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls
    )
    return message


