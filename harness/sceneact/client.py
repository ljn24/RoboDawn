"""Native RoboDawn Chat Completions requests with bounded retries and audit logs."""
from email.utils import parsedate_to_datetime
import http.client
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from harness.agent.llm_client import ChatClient, ChatResult, USER_AGENT, thinking_fields
from harness.agent.mllm_agent import AgentConfig


class ModelRequestError(RuntimeError):
    pass


def evaluation_model_config(baseline):
    """Keep frozen evidence unchanged; describe the actual RoboDawn request separately."""
    defaults = AgentConfig()
    config = {key: baseline[key] for key in (
        'model', 'base_url', 'use_env_proxy', 'api_key_env', 'timeout_s',
        'total_timeout_s', 'max_retries', 'retry_backoff_s', 'retry_statuses', 'fatal_statuses')}
    config.update(api='chat_completions', temperature=defaults.temperature, max_tokens=defaults.max_tokens,
                  request=dict(thinking_fields(baseline['model']),
                               reasoning_effort=baseline['request']['reasoning']['effort']))
    return config


def retry_delay(value):
    if value is None:
        return 0.
    try:
        return max(0., float(value))
    except ValueError:
        try:
            return max(0., parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.


class MatchedChatClient(ChatClient):
    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.request_index = 0
        self.records = []
        if config['api'] != 'chat_completions' or config.get('stream', False):
            raise ValueError('RoboDawn uses non-streaming Chat Completions')
        self.key = os.environ.get(config['api_key_env'], '').strip()
        if not self.key:
            raise ValueError(f'missing {config["api_key_env"]}')
        super().__init__(model=config['model'], base_url=config['base_url'], api_key=self.key,
                         timeout_s=config['timeout_s'], temperature=config['temperature'],
                         max_tokens=config['max_tokens'], extra_body=config['request'])
        self.total_usage = {}
        handlers = [] if config.get('use_env_proxy', False) else [urllib.request.ProxyHandler({})]
        self.opener = urllib.request.build_opener(*handlers)

    def chat(self, messages, tag=''):
        cfg = self.config
        body = self._body(messages)
        self.request_index += 1
        start = time.monotonic()
        deadline = start + cfg['total_timeout_s']
        record = dict(request_index=self.request_index, tag=tag, model=cfg['model'], api=cfg['api'],
                      parameters={k: v for k, v in body.items() if k != 'messages'},
                      attempts=[], transport_ok=False, usage=None,
                      image_count=sum(p.get('type') == 'image_url' for m in body['messages']
                                      if isinstance(m['content'], list) for p in m['content']))
        directory = self.output / 'model_inputs'
        directory.mkdir(parents=True, exist_ok=True)
        input_path = directory / f'{self.request_index:04d}.json'
        input_path.write_text(json.dumps(body, ensure_ascii=False), encoding='utf-8')
        record['input_file'] = str(input_path.relative_to(self.output))
        payload = json.dumps(body).encode()
        try:
            for attempt in range(cfg['max_retries'] + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModelRequestError('model request retry time budget exhausted')
                detail = dict(attempt=attempt + 1)
                record['attempts'].append(detail)
                attempt_start = time.monotonic()
                request = urllib.request.Request(self.base_url + '/chat/completions', data=payload,
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key,
                             'User-Agent': USER_AGENT}, method='POST')
                try:
                    with self.opener.open(request, timeout=min(cfg['timeout_s'], remaining)) as response:
                        detail['http_status'] = response.status
                        detail['request_id'] = response.headers.get('x-request-id')
                        raw = json.load(response)
                    error = raw.get('error') if isinstance(raw, dict) else None
                    if error:
                        transient = (isinstance(error, dict) and error.get('type') == 'new_api_error'
                                     and str(error.get('message', '')).startswith((
                                         'The upstream stream ended before the response was complete.',
                                         'The model service is temporarily unavailable. Please try again later.')))
                        if transient:
                            raise ConnectionError(f'model endpoint returned error: {error}')
                        raise ModelRequestError(str(error).replace(self.key, '[REDACTED]'))
                    record['transport_ok'] = True
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
                    message = f'{type(exc).__name__}: {exc}'
                    retryable, delay = True, cfg['retry_backoff_s'] * 2 ** attempt
                    if isinstance(exc, urllib.error.HTTPError):
                        detail['http_status'] = exc.code
                        try:
                            message += ': ' + exc.read().decode(errors='replace')
                            delay = max(delay, retry_delay(exc.headers.get('Retry-After')))
                        finally:
                            exc.close()
                        retryable = exc.code in cfg['retry_statuses']
                    detail['error'] = message.replace(self.key, '[REDACTED]')
                    if not retryable or attempt == cfg['max_retries'] or delay >= deadline - time.monotonic():
                        raise ModelRequestError(detail['error']) from None
                    detail['retry_delay_s'] = delay
                    time.sleep(delay)
                finally:
                    detail['wall_s'] = time.monotonic() - attempt_start
            choice = raw['choices'][0]
            record.update(response_model=raw.get('model'), response_id=raw.get('id'),
                          usage=raw.get('usage'), finish_reason=choice.get('finish_reason'))
            usage = raw.get('usage') or {}
            for source, key in [('prompt_tokens', 'input_tokens'), ('completion_tokens', 'output_tokens'),
                                ('total_tokens', 'total_tokens')]:
                if isinstance(usage.get(source), int):
                    self.total_usage[key] = self.total_usage.get(key, 0) + usage[source]
            for key, field in [('reasoning_tokens', 'completion_tokens_details'), ('cached_tokens', 'prompt_tokens_details')]:
                count = (usage.get(field) or {}).get(key)
                if key == 'cached_tokens' and count is None:
                    count = usage.get('prompt_cache_hit_tokens')
                if isinstance(count, int):
                    self.total_usage[key] = self.total_usage.get(key, 0) + count
            # Match native ChatClient: pass text through to RoboDawn's tolerant JSON parser.
            text = choice['message'].get('content') or ''
            if isinstance(text, list):
                text = ''.join(p.get('text', '') for p in text if isinstance(p, dict))
            record.update(response=text, reasoning=choice['message'].get('reasoning_content') or '')
            return ChatResult(text=text, latency_s=time.monotonic()-start,
                              usage=usage, raw=raw, attempts=len(record['attempts']))
        except Exception as exc:
            record['error'] = str(exc).replace(self.key, '[REDACTED]')
            if isinstance(exc, ModelRequestError):
                raise
            raise ModelRequestError(record['error']) from None
        finally:
            record['wall_s'] = time.monotonic() - start
            self.records.append(record)
            with (self.output / 'model.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
