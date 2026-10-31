"""Offline checks for pairing, transport, removed native limits and statistics."""
import io
import copy
import json
import os
from pathlib import Path
import tempfile
import sys
import signal
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from PIL import Image

from harness.agent.llm_client import ChatClient, ChatResult
from harness.agent.mllm_agent import AgentConfig, MLLMDiscreteAgent
from harness.agent.demos import build_demo
from harness.agent.prompts import DEMO_NOTE, PRIMER_NOTE, parse_agent_json
from harness.core.env import View
from harness.sceneact.agent import MatchedAgent
from harness.sceneact.client import ModelRequestError, MatchedChatClient, evaluation_model_config
from harness.sceneact.episode import EpisodeControl, EpisodeFinished, OfficialEvaluationError
from harness.sceneact.execution import install_step_budget, set_motion_speed
from harness.sceneact.run_eval import aggregate, select_jobs, REFERENCE, ROOT
from harness.sceneact.timing import Timings
from harness.sceneact import run_eval


class FakeEnv:
    def __init__(self, stop=None):
        self.control = EpisodeControl(10, .004, .2, 10)
        self.planning_wall_s = self.evaluation_wall_s = self.video_capture_s = self.physics_wall_s = 0.
        self.stop = stop

    @property
    def sim_time_s(self):
        return self.control.physics_steps * .004

    @property
    def success(self):
        return self.control.success

    @property
    def steps_used(self):
        return self.control.primitive_executions

    @property
    def step_limit(self):
        return self.control.primitive_limit

    @property
    def budget_exhausted(self):
        return self.control.status is not None

    def state(self):
        arm = dict(position_cm=[0, 0, 90], gripper_real=1., approach=[0, 0, -1], finger_axis=[0, 1, 0])
        return dict(steps_used=self.steps_used, step_limit=self.step_limit,
                    table_z_cm=74., left=arm, right=arm, success=False)

    def prompt_context(self):
        return {'table_z': 74}

    def observe(self):
        return dict(state=self.state())

    def views(self, obs, cameras):
        names = [name for camera in cameras
                 for name in (('left_camera', 'right_camera') if camera == 'wrist' else (camera,))]
        return [View(name, Image.new('RGB', (2, 2)), name) for name in names]

    def execute(self, command):
        self.control.begin_action()
        self.control.physics_steps += 1
        self.control.primitive_executions += 1
        return dict(command=command.text(), kind=command.kind, ok=True, after=self.state())

    def should_stop(self):
        return self.stop


class BudgetTest(unittest.TestCase):
    def test_failures_without_physics_do_not_spend_execution_budget(self):
        c = EpisodeControl(400, .004, .2, 10)
        for _ in range(100):
            c.begin_action()
        self.assertIsNone(c.status)
        self.assertEqual(c.primitive_limit, 40)
        self.assertEqual(c.physics_step_limit, 20000)
        c.primitive_executions = 40
        self.assertEqual(c.status, 'primitive_budget')

    def test_final_primitive_can_finish_but_physics_boundary_interrupts(self):
        c = EpisodeControl(10, .004, .2, 10)
        c.primitive_executions = 1
        c.after_step(False)
        c.physics_steps = c.physics_step_limit
        with self.assertRaises(EpisodeFinished) as caught:
            c.after_step(False)
        self.assertEqual(caught.exception.status, 'simulation_budget')

    def test_success_wins_at_boundary(self):
        c = EpisodeControl(10, .004, .2, 10)
        c.physics_steps = c.physics_step_limit
        with self.assertRaises(EpisodeFinished) as caught:
            c.after_step(True)
        self.assertEqual(caught.exception.status, 'official_success')


class ExecutionTest(unittest.TestCase):
    def adapter(self, checker):
        task = SimpleNamespace(scene=SimpleNamespace(step=unittest.mock.Mock()),
                               check_success=checker, _update_render=lambda: None, eval_success=False)
        adapter = SimpleNamespace(env=task, control=EpisodeControl(1, .1, .2, 10),
                                  physics_wall_s=0., evaluation_wall_s=0.)
        install_step_budget(adapter)
        return adapter

    def test_native_rechecks_do_not_duplicate_stateful_evaluator(self):
        checker = unittest.mock.Mock(return_value=False)
        adapter = self.adapter(checker)
        adapter.env.scene.step()
        adapter.env.check_success()
        adapter.env.check_success()
        self.assertEqual(checker.call_count, 1)
        with self.assertRaises(EpisodeFinished) as caught:
            adapter.env.scene.step()
        self.assertEqual(caught.exception.status, 'simulation_budget')
        self.assertEqual(checker.call_count, 2)
        with self.assertRaises(EpisodeFinished):
            adapter.env.scene.step()
        self.assertEqual(adapter.control.physics_steps, 2)

    def test_evaluator_failure_is_not_an_action_failure(self):
        adapter = self.adapter(unittest.mock.Mock(side_effect=ValueError('bad checker')))
        with self.assertRaises(OfficialEvaluationError):
            adapter.env.scene.step()
        self.assertEqual(adapter.control.physics_steps, 1)

    def test_curobo_receives_actual_speed_parameter(self):
        received = []
        mg = SimpleNamespace(plan_single=lambda start, goal, cfg: received.append(cfg.time_dilation_factor) or 'trajectory')
        callback = unittest.mock.Mock()
        set_motion_speed(mg, .5, callback)
        result = mg.plan_single('start', 'goal', SimpleNamespace(time_dilation_factor=None))
        self.assertEqual(result, 'trajectory')
        self.assertEqual(received, [.5])
        self.assertEqual(callback.call_args.args[1], .5)


class PairingTest(unittest.TestCase):
    def test_reference_is_54_percent_and_unique(self):
        ref = json.loads(REFERENCE.read_text())
        self.assertEqual(len(ref['jobs']), 250)
        self.assertEqual(sum(j['baseline']['success'] for j in ref['jobs']), 135)
        self.assertEqual(len({(j['task'], j['episode_index']) for j in ref['jobs']}), 250)
        self.assertEqual(ref['budget'], dict(motion_speed=.5, sim_seconds_per_official_step=.2, primitive_budget_divisor=10))
        self.assertEqual(ref['model']['request'], dict(reasoning=dict(effort='low')))

    def test_budget_and_timing_are_exact_frozen_copies(self):
        import hashlib
        ref = json.loads(REFERENCE.read_text())
        for local, original in [('episode.py', 'sceneact/core/episode.py'), ('timing.py', 'sceneact/runtime/timing.py')]:
            self.assertEqual(hashlib.sha256((ROOT / 'harness/sceneact' / local).read_bytes()).hexdigest(),
                             ref['sceneact_source_hashes'][original])

    def test_job_selection_preserves_pair_identity(self):
        ref = json.loads(REFERENCE.read_text())
        ids = [ref['jobs'][13]['id'], ref['jobs'][2]['id']]
        jobs = select_jobs(ref, ','.join(ids))
        self.assertEqual([j['id'] for j in jobs], ids[::-1])
        with self.assertRaises(ValueError):
            select_jobs(ref, ids[0] + ',' + ids[0])

    def test_aggregate_denominator_and_missing_metrics(self):
        ref = json.loads(REFERENCE.read_text())
        jobs = select_jobs(ref, ','.join(j['id'] for j in ref['jobs'][:2]))
        r = dict(jobs[0], success=True, evaluation_completed=True, status='official_success', decisions=7,
                 budget=dict(primitive_executions=3), tokens=dict(input_tokens=10))
        with tempfile.TemporaryDirectory() as directory:
            summary = aggregate(Path(directory), jobs, ref, {r['id']: r})
        self.assertEqual(summary['success_rate'], .5)
        self.assertEqual(summary['averages']['successful']['primitive_executions'], dict(mean=3., n=1))
        self.assertEqual(summary['averages']['successful']['reasoning_tokens'], dict(mean=None, n=0))


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.baseline = json.loads(REFERENCE.read_text())['model']
        self.config = evaluation_model_config(self.baseline)
        with patch.dict(os.environ, DEEPSEEK_API_KEY='private-test-key'):
            self.client = MatchedChatClient(self.config, self.tmp.name)
        self.messages = [dict(role='system', content='system'),
                         dict(role='user', content=[dict(type='text', text='primer'),
                                                   dict(type='image_url', image_url=dict(url='data:image/png;base64,AA=='))]),
                         dict(role='assistant', content='ack'), dict(role='user', content='now')]

    def response(self, finish_reason='stop', content='{"commands":["wait"]}'):
        data = dict(model='deepseek-flash', id='reply',
                    usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                               completion_tokens_details=dict(reasoning_tokens=3)),
                    choices=[dict(finish_reason=finish_reason, message=dict(content=content))])
        response = io.BytesIO(json.dumps(data).encode())
        response.status, response.headers = 200, {}
        return response

    def test_request_matches_native_body_with_explicit_low_effort(self):
        defaults = AgentConfig()
        native = ChatClient(model=self.config['model'], base_url=self.config['base_url'], api_key='test',
                            temperature=defaults.temperature, max_tokens=defaults.max_tokens,
                            extra_body={'reasoning_effort': 'low'})
        body = self.client._body(self.messages)
        self.assertEqual(body, native._body(self.messages))
        self.assertEqual(body['messages'], self.messages)
        self.assertEqual(body['temperature'], 0.)
        self.assertEqual(body['max_tokens'], 8000)
        self.assertEqual(body['thinking'], {'type': 'enabled'})
        self.assertEqual(body['reasoning_effort'], 'low')
        self.assertNotIn('response_format', body)
        self.assertEqual(self.baseline, json.loads(REFERENCE.read_text())['model'])

    def test_retry_then_usage_recorded(self):
        failure = urllib.error.HTTPError('https://invalid', 429, 'rate', {'Retry-After': '2'}, io.BytesIO(b'limited'))
        with patch.object(self.client.opener, 'open', side_effect=[failure, self.response()]) as op, patch('time.sleep'):
            result = self.client.chat(self.messages)
        self.assertEqual(op.call_args.args[0].full_url, self.config['base_url'] + '/chat/completions')
        self.assertEqual(json.loads(op.call_args.args[0].data), self.client._body(self.messages))
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.usage['prompt_tokens'], 10)
        self.assertEqual(self.client.total_usage['input_tokens'], 10)
        self.assertEqual(self.client.total_usage['output_tokens'], 5)
        self.assertEqual(self.client.total_usage['reasoning_tokens'], 3)
        record = json.loads((Path(self.tmp.name) / 'model.jsonl').read_text())
        self.assertEqual(record['image_count'], 1)
        self.assertEqual(record['attempts'][0]['http_status'], 429)

    def test_fatal_api_error_no_retry_and_secret_redacted(self):
        failure = urllib.error.HTTPError('https://invalid', 402, 'payment', {}, io.BytesIO(b'private-test-key'))
        with patch.object(self.client.opener, 'open', side_effect=failure) as op:
            with self.assertRaises(ModelRequestError):
                self.client.chat(self.messages)
        self.assertEqual(op.call_count, 1)
        self.assertNotIn('private-test-key', (Path(self.tmp.name) / 'model.jsonl').read_text())

    def test_truncated_text_is_passed_to_native_parser(self):
        partial = '{"commands":["wait"'
        with patch.object(self.client.opener, 'open', return_value=self.response('length', partial)):
            result = self.client.chat(self.messages)
        self.assertEqual(result.text, partial)
        with self.assertRaises(ValueError):
            parse_agent_json(result.text)

    def test_unicode_model_output_is_written_as_utf8(self):
        response = self.response()
        raw = json.loads(response.getvalue())
        raw['choices'][0]['message']['content'] = '{"scene":"\\u2248","commands":["wait"]}'
        response = io.BytesIO(json.dumps(raw).encode())
        response.status, response.headers = 200, {}
        with patch.object(self.client.opener, 'open', return_value=response):
            result = self.client.chat(self.messages)
        record = json.loads((Path(self.tmp.name) / 'model.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(record['response'], result.text)


class WorkerLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'workspace').mkdir()
        self.job = dict(id='test_job', task='test', seed=1)
        self.plan = dict(robotwin_root=str(self.root), python=sys.executable, process_timeout_s=.1)
        self.devices = {0: dict(uuid='test-uuid', pci='0000:01:00.0')}
        run_eval.STOP_REQUESTED.clear()
        self.addCleanup(run_eval.STOP_REQUESTED.clear)

    def test_timeout_kills_group_and_marks_incomplete(self):
        process = SimpleNamespace(pid=12345, wait=unittest.mock.Mock(side_effect=[subprocess.TimeoutExpired('worker', .1), 0]))
        with patch.object(run_eval.subprocess, 'Popen', return_value=process) as spawn, patch.object(run_eval.os, 'killpg') as kill:
            result = run_eval.run_job(self.job, (0, 2), self.plan, self.root, self.devices)
        self.assertEqual(kill.call_args_list, [unittest.mock.call(12345, signal.SIGTERM),
                                              unittest.mock.call(12345, signal.SIGKILL)])
        self.assertEqual(result['status'], 'process_timeout')
        self.assertFalse(result['evaluation_completed'])
        self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'test-uuid')
        self.assertEqual(spawn.call_args.kwargs['env']['OMP_NUM_THREADS'], '1')
        self.assertEqual(spawn.call_args.kwargs['env']['PYTHONUTF8'], '1')
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        self.assertFalse(run_eval.ACTIVE_PROCESSES)
        self.assertFalse((self.root / 'jobs/test_job/tmp').exists())

    def test_cancellation_cannot_launch_new_worker(self):
        run_eval.STOP_REQUESTED.set()
        with patch.object(run_eval.subprocess, 'Popen') as spawn:
            result = run_eval.run_job(self.job, (0, 0), self.plan, self.root, self.devices)
        spawn.assert_not_called()
        self.assertEqual(result['status'], 'interrupted')


class AgentTest(unittest.TestCase):
    def test_prompts_and_mixed_command_feedback_match_native_primer_only_loop(self):
        replies = ['invalid', '{"commands":[]}', '{"commands":["bad"]}',
                   '{"commands":["done"]}', '{"commands":["wait","bad"]}', '{"commands":["wait"]}']
        def client_for(messages):
            pending = iter(replies)
            def chat(sent, **kwargs):
                messages.append(copy.deepcopy(sent))
                return ChatResult(next(pending), 0)
            return SimpleNamespace(chat=chat)

        reference = json.loads(REFERENCE.read_text())
        actual, expected = [], []
        with tempfile.TemporaryDirectory() as directory:
            matched = MatchedAgent(ROOT, client_for(actual), reference, directory)
            env = FakeEnv()
            env.control.primitive_limit = 2
            report = {}
            matched.run(env, 'test', Timings(), report)
            # Upstream's loader requires a task demo; supply its existing primer only
            # to exercise the unmodified policy loop in the zero-shot setting.
            with patch('harness.agent.mllm_agent.load_demos',
                       return_value=[build_demo(ROOT / 'demos/robotwin2/primer')]):
                native = MLLMDiscreteAgent(AgentConfig(profile=matched.cfg.profile, demo_bank=None),
                                          client_for(expected))
            native_env = FakeEnv()
            native_env.control.primitive_limit = 2
            native.run_episode(native_env, 'test')
            # Only the system's demonstration paragraph differs from the native loop.
            for messages in expected:
                messages[0]['content'] = messages[0]['content'].replace(DEMO_NOTE, PRIMER_NOTE)
            self.assertEqual(actual, expected)
            self.assertEqual(report['decisions'], 6)
            self.assertEqual(report['invalid_decisions'], 1)
            self.assertEqual(env.steps_used, native_env.steps_used)
            self.assertEqual((Path(directory) / 'system_prompt.txt').read_text(), expected[0][0]['content'])

    def run_agent(self, replies, stop=None):
        ref = json.loads(REFERENCE.read_text())
        reply_iter = iter(replies)
        client = SimpleNamespace(chat=lambda *a, **k: ChatResult(next(reply_iter), 0))
        env, report = FakeEnv(stop), {}
        with tempfile.TemporaryDirectory() as directory:
            agent = MatchedAgent(ROOT, client, ref, directory)
            self.assertEqual(len(agent.demo_msgs), 2)
            self.assertEqual(agent.cfg.cameras, ['agent_camera', 'top_camera', 'wrist'])
            agent.run(env, 'test', Timings(), report)
        return report, env

    def test_old_turn_done_and_parse_limits_removed(self):
        replies = ['invalid'] * 5 + ['{"commands":["done"]}'] * 46 + ['{"commands":["wait"]}']
        report, env = self.run_agent(replies)
        self.assertEqual(report['decisions'], 52)
        self.assertEqual(report['status'], 'primitive_budget')
        self.assertEqual(report['image_views'], 52)
        self.assertEqual(report['image_requests'], 51)
        self.assertEqual(report['current_images'], 208)
        self.assertEqual(env.control.primitive_attempts, 52)
        self.assertEqual(env.control.primitive_executions, 1)

    def test_object_fell_retained_without_physical_execution(self):
        report, env = self.run_agent(['{"commands":["done"]}'], stop='object_fell')
        self.assertEqual(report['status'], 'object_fell')
        self.assertTrue(report['evaluation_completed'])
        self.assertEqual(env.control.primitive_executions, 0)


if __name__ == '__main__':
    unittest.main()
