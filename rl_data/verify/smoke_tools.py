"""Check a direct provider's tool call and continuation without running shell code."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=os.getenv('VERIFY_ROLLOUT_MODEL', 'gemini/gemini-3-flash-preview'))
    args = parser.parse_args()
    tools = [{'type': 'function', 'function': {'name': 'bash', 'description': 'Run a shell command.', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
    messages = [{'role': 'user', 'content': 'Call bash exactly once with command printf ok. Do not answer until you receive the tool result.'}]
    response = complete(messages, model=args.model, tools=tools, max_output_tokens=8192, timeout=120)
    msg = response.choices[0].message
    if not msg.tool_calls or len(msg.tool_calls) != 1:
        raise RuntimeError('Expected exactly one tool call')
    call = msg.tool_calls[0]
    if call.function.name != 'bash' or 'command' not in json.loads(call.function.arguments):
        raise RuntimeError('Invalid bash call')
    messages.append(msg.model_dump(exclude_none=True))
    messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': 'ok'})
    messages.append({'role': 'user', 'content': 'The test is complete. Reply with ok and do not call any more tools.'})
    reply = complete(messages, model=args.model, tools=tools, max_output_tokens=8192, timeout=120)
    if not reply.choices[0].message.content:
        raise RuntimeError('No tool continuation response')
    print('TOOL SMOKE OK:', args.model, '(tool arguments and continuation accepted)')


if __name__ == '__main__':
    main()
