import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import litellm
from rl_data.verify import llm_client as client


class DirectProviderTests(unittest.TestCase):
    def test_credentials_follow_requested_model(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'openai-test', 'GEMINI_API_KEY': 'google-test'}, clear=True):
            self.assertEqual(client.load_api_key('openai/gpt-5'), 'openai-test')
            self.assertEqual(client.load_api_key('gemini/gemini-3.1-pro-preview'), 'google-test')

    def test_credential_file_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            key = Path(folder) / 'key'
            key.write_text('file-key')
            key.chmod(0o644)
            with patch.dict(os.environ, {'GEMINI_API_KEY_FILE': str(key)}, clear=True):
                with self.assertRaises(client.ProviderAuthError):
                    client.load_api_key('gemini/test')
                key.chmod(0o600)
                self.assertEqual(client.load_api_key('gemini/test'), 'file-key')

    def test_reject_gateway_alias_and_missing_prefix(self):
        for model in ['gpt-5', 'openai/gpt-5-6-sol-genai-responses', 'metagen/test']:
            with self.assertRaises(ValueError):
                client.complete([], model=model, api_key='test')

    def test_provider_request_does_not_include_gateway(self):
        with patch.object(litellm, 'completion', return_value=object()) as call:
            client.complete([], model='gemini/test', api_key='test', tools=[{'type': 'function'}], response_format={'type': 'json_object'})
            self.assertNotIn('api_base', call.call_args.kwargs)
            self.assertEqual(call.call_args.kwargs['max_tokens'], 4096)
            self.assertIn('tools', call.call_args.kwargs)
            client.complete([], model='openai/gpt-5', api_key='test')
            self.assertEqual(call.call_args.kwargs['max_completion_tokens'], 4096)

    def test_transient_retries_but_invalid_requests_do_not(self):
        rate = litellm.RateLimitError(message='busy', model='test', llm_provider='gemini')
        with patch.object(litellm, 'completion', side_effect=[rate, 'ok']) as call, patch.object(client.time, 'sleep'):
            self.assertEqual(client.complete([], model='gemini/test', api_key='test'), 'ok')
            self.assertEqual(call.call_count, 2)
        bad = litellm.BadRequestError(message='bad', model='test', llm_provider='gemini')
        with patch.object(litellm, 'completion', side_effect=bad) as call:
            with self.assertRaises(litellm.BadRequestError):
                client.complete([], model='gemini/test', api_key='test')
            self.assertEqual(call.call_count, 1)

    def test_auth_failure_stops_without_exposing_provider_error(self):
        exc = litellm.AuthenticationError(message='secret-key', model='test', llm_provider='gemini')
        with patch.object(litellm, 'completion', side_effect=exc) as call:
            with self.assertRaises(client.ProviderAuthError) as error:
                client.complete([], model='gemini/test', api_key='test')
            self.assertNotIn('secret-key', str(error.exception))
            self.assertEqual(call.call_count, 1)


class ToolHistoryTests(unittest.TestCase):
    def test_trim_preserves_origin_of_retained_tool_result(self):
        from rl_data.verify.rollout import trim
        origin = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c1'}], 'provider_specific_fields': {'signature': 'test'}}
        messages = [{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': 'task'}]
        messages += [{'role': 'user', 'content': 'old'}] * 5
        messages += [origin, {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}]
        messages += [{'role': 'user', 'content': 'recent'}] * 11
        result = trim(messages)
        self.assertEqual(result[3], origin)
        self.assertEqual(result[4]['tool_call_id'], 'c1')
