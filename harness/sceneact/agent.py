"""RoboDawn zero-shot loop with SceneAct termination and accounting."""
import json
from dataclasses import asdict
from pathlib import Path
import time

import yaml

from harness.agent.demos import build_demo, demo_messages, save_demo
from harness.agent.llm_client import text_part
from harness.agent.memory import AgentMemory
from harness.agent.mllm_agent import AgentConfig, MLLMDiscreteAgent, _update_grasp_facts
from harness.agent.prompts import PRIMER_NOTE, parse_agent_json, system_prompt, turn_text
from harness.core.commands import parse_command_list


class MatchedAgent(MLLMDiscreteAgent):
    def __init__(self, root, client, reference, output):
        self.client, self.output = client, Path(output)
        profile = yaml.safe_load((root / 'harness/configs/robotwin2_profile.yaml').read_text(encoding='utf-8'))
        # Keep the native AgentConfig camera selection and benchmark profile.
        self.cfg = AgentConfig(model=reference['model']['model'], profile=profile, demo_bank=None,
                               max_turns=None, done_limit=None, consecutive_parse_failures_allowed=None,
                               timeout_s=reference['model']['timeout_s'])
        (self.output / 'agent_config.json').write_text(json.dumps(asdict(self.cfg), indent=2), encoding='utf-8')
        primer = build_demo(root / 'demos/robotwin2/primer')
        self.demo_msgs = demo_messages([primer])
        save_demo(primer, self.output / 'demo_primer')

    def run(self, env, instruction, timing, report):
        cfg = self.cfg
        memory = AgentMemory(use_history=cfg.memory_history, use_scratchpad=cfg.memory_scratchpad,
                             history_turns=cfg.history_turns)
        prompt = system_prompt(cfg.profile, cfg.max_commands_per_turn, cfg.memory_scratchpad,
                               context=env.prompt_context(), demo_note=PRIMER_NOTE)
        (self.output / 'system_prompt.txt').write_text(prompt, encoding='utf-8')
        last_results, grasp_facts = [], {}
        report.update(decisions=0, image_requests=0, image_views=0, current_images=0,
                      model_calls=0, invalid_decisions=0, status='running', success=False,
                      evaluation_completed=False, agent_mode='robodawn_closed_loop', shots=0,
                      command_primer=True, max_commands_per_turn=cfg.max_commands_per_turn)
        while env.control.status is None:
            with timing.measure('observation'):
                obs = env.observe()
                state = obs['state']
                images, captions, saved = self._images_for_turn(env, obs)
            report['decisions'] += 1
            turn = report['decisions']
            with timing.measure('logging'):
                for name, data in saved.items():
                    (self.output / f'turn{turn:03d}_{name}.png').write_bytes(data)
            text = turn_text(turn, instruction, state, last_results, memory.render(), captions,
                             grasp_facts=grasp_facts)
            messages = [dict(role='system', content=prompt)] + self.demo_msgs + [
                dict(role='user', content=images + [text_part(text)])]
            report['model_calls'] += 1
            report['image_views'] += int(bool(images))
            report['image_requests'] = max(0, report['image_views']-1)
            report['current_images'] += len(images)
            # RoboDawn has no separate initialization request. All model time is llm_decision.
            with timing.measure('llm_decision'):
                reply = self.client.chat(messages, tag=f'turn{turn}')
            record = dict(index=turn-1, state=state, prompt=text, reply=reply.text, commands=[], results=[])
            try:
                parsed = parse_agent_json(reply.text)
                raw = parsed.get('commands') or []
                if isinstance(raw, str):
                    raw = [raw]
                commands, errors = parse_command_list(raw)
            except Exception as exc:
                env.control.begin_action()
                report['invalid_decisions'] += 1
                last_results = [dict(command='(unparseable reply)', ok=False,
                    note=f'your reply was not a valid JSON object ({exc}); reply with the JSON object only')]
                record['parse_error'] = str(exc)
                record['output_valid'] = False
            else:
                record['output_valid'] = True
                memory.update_scratchpad(parsed.get('memory'))
                record.update(scene=parsed.get('scene'), progress=parsed.get('progress'),
                              plan=parsed.get('plan'), memory=parsed.get('memory'), command_errors=errors)
                commands = commands[:cfg.max_commands_per_turn]
                record['commands'] = [cmd.text() for cmd in commands]
                last_results = []
                for cmd in commands:
                    if env.control.status:
                        break
                    if cmd.kind == 'done':
                        env.control.begin_action()
                        last_results.append(dict(command=cmd.text(), kind='done', arm=None, ok=False,
                            note=('NOT finished: the benchmark checker has not registered success, so the episode continues. '
                                  'Re-read the instruction; typical reasons: the object is not where/how the task wants it '
                                  '(wrong spot, not high enough, not far enough to the side, wrong orientation), a gripper '
                                  'must be opened at the end, or a second object/arm is still required. Keep acting.')))
                        break
                    before = (env.planning_wall_s, env.evaluation_wall_s, env.video_capture_s)
                    start = time.monotonic()
                    try:
                        result = env.execute(cmd)
                    finally:
                        elapsed = time.monotonic()-start
                        after = (env.planning_wall_s, env.evaluation_wall_s, env.video_capture_s)
                        deltas = [b-a for a, b in zip(before, after)]
                        for name, value in zip(('planning', 'evaluation', 'logging', 'execution'),
                                               deltas + [max(0., elapsed-sum(deltas))]):
                            timing.seconds[name] = timing.seconds.get(name, 0.) + value
                    last_results.append(result)
                    _update_grasp_facts(grasp_facts, cmd, result)
                for err in errors:
                    last_results.append(dict(command='(invalid)', kind='invalid', arm=None, ok=False, note=err))
                memory.record_turn(turn, record['commands'], last_results, env.state())
            env.control.after_decision(dict(ok=any(r['ok'] for r in last_results),
                                            executed=[dict(result=r) for r in last_results]))
            record.update(results=last_results, budget=env.control.summary())
            with timing.measure('logging'):
                with (self.output / 'trace.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                (self.output / 'memory.json').write_text(json.dumps(memory.to_json(), ensure_ascii=False), encoding='utf-8')
            if env.control.status:
                break
            if 'parse_error' in record:
                # Native RoboDawn requests another reply before evaluating early stops.
                continue
            with timing.measure('evaluation'):
                stop = env.should_stop()
            if stop:
                report.update(status=stop, evaluation_completed=True)
                break
        if env.control.status:
            report.update(status=env.control.status, success=env.control.success, evaluation_completed=True)
