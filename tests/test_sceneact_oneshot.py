"""One-shot selection, frozen assets and paired-budget regression checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness.agent.demos import build_demo, load_saved_demo, demo_messages
from harness.agent.llm_client import ChatResult
from harness.agent.prompts import DEMO_NOTE, PRIMER_NOTE
from harness.sceneact.agent import MatchedAgent
from harness.sceneact.run_eval import ROOT, prepare_one_shot_jobs, source_hashes
from harness.sceneact.timing import Timings
from test_sceneact_protocol import FakeEnv


class OneShotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'harness/configs/sceneact/one_shot_20.json').read_text())
        cls.reference = json.loads((ROOT / 'harness/configs/sceneact/reference_actions2x.json').read_text())
        cls.jobs = prepare_one_shot_jobs(cls.reference, None, cls.config)

    def test_pairing_and_budget(self):
        self.assertEqual(len(self.jobs), 100)
        originals = {j['id']: j for j in self.reference['jobs']}
        for task in self.config['tasks']:
            self.assertEqual({j['episode_index'] for j in self.jobs if j['task'] == task}, set(range(5)))
        for job in self.jobs:
            for key in ('seed', 'instruction', 'step_limit', 'task_config'):
                self.assertEqual(job[key], originals[job['id']][key])
            self.assertEqual(job['demo_path'], f"demos/robotwin2/expert/{job['task']}")
        self.assertEqual(self.reference['budget'], dict(sim_seconds_per_official_step=.2,
                                                       primitive_budget_divisor=5., motion_speed=.5))

    def test_subset_rejects_tasks_outside_preset(self):
        other = next(j for j in self.reference['jobs'] if j['task'] not in self.config['tasks'])
        with self.assertRaises(ValueError):
            prepare_one_shot_jobs(self.reference, other['id'], self.config)
        self.assertEqual(prepare_one_shot_jobs(self.reference, self.jobs[0]['id'], self.config), self.jobs[:1])

    def test_missing_first_demo_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                prepare_one_shot_jobs(self.reference, None, self.config, Path(directory))

    def test_fixed_first_messages_and_no_selection(self):
        job = next(j for j in self.jobs if j['task'] == 'place_mouse_pad')
        captured = []
        def chat(messages, **kwargs):
            captured.append(copy.deepcopy(messages))
            return ChatResult('{"commands":["wait"]}', 0)
        with tempfile.TemporaryDirectory() as directory, patch(
                'harness.agent.demos.select_demo', side_effect=AssertionError('auto-selection forbidden')):
            agent = MatchedAgent(ROOT, SimpleNamespace(chat=chat), self.reference, directory, job=job)
            report = {}
            agent.run(FakeEnv(), 'paired instruction', Timings(), report)
            demo = load_saved_demo(ROOT / job['demo_path'])
            expected = demo_messages([build_demo(ROOT / 'demos/robotwin2/primer'), demo])
            self.assertEqual(captured[0][1:3], expected)
            self.assertIn(DEMO_NOTE, captured[0][0]['content'])
            self.assertNotIn(PRIMER_NOTE, captured[0][0]['content'])
            self.assertEqual(report['shots'], 1)
            self.assertEqual(report['demo_source'], demo.source)
            self.assertTrue((Path(directory) / 'demo_task/demo.json').is_file())

    def test_expert_assets_are_fingerprinted(self):
        sources = source_hashes(ROOT, shots=1)
        for job in self.jobs:
            self.assertIn(job['demo_path'] + '/demo.json', sources)
        self.assertNotIn(self.jobs[0]['demo_path'] + '/demo.json', source_hashes(ROOT))
