"""One isolated matched episode; invoked by run_eval, not the native runner."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from .client import ModelRequestError, MatchedChatClient, evaluation_model_config
from .timing import Timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--render-device', required=True)
    args = parser.parse_args()
    # Resolve before importing RoboTwin, which changes the working directory.
    output, job_path, ref_path = args.output.resolve(), args.job.resolve(), args.reference.resolve()
    root = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=True)
    job, reference = json.loads(job_path.read_text(encoding='utf-8')), json.loads(ref_path.read_text(encoding='utf-8'))
    report = dict(job, success=False, evaluation_completed=False, status='worker_error',
                  devices=dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), worker_pid=os.getpid()))
    timing, env, client = Timings(), None, None
    start = None
    cold_start = time.monotonic()
    try:
        from .env import MatchedEnv
        from .agent import MatchedAgent
        model_config = evaluation_model_config(reference['model'])
        report['model_config'] = model_config
        client = MatchedChatClient(model_config, output)
        agent = MatchedAgent(root, client, reference, output, job=job)
        env = MatchedEnv(job, reference, args.render_device)
        report['cold_start_s'] = time.monotonic()-cold_start
        reset = time.monotonic()
        episode = env.reset_episode()
        report['reset_s'] = time.monotonic()-reset
        report['devices'].update(env.devices())
        start = time.monotonic()
        agent.run(env, episode.instruction, timing, report)
    except Exception as exc:
        report.update(status='model_api_error' if isinstance(exc, ModelRequestError) else 'worker_error',
                      error=str(exc), traceback=traceback.format_exc(), evaluation_completed=False, success=False)
    finally:
        if env is not None and hasattr(env, 'control'):
            try:
                with timing.measure('logging'):
                    env._record_frame('final')
                    report['video'] = dict(file=env.save_video(output / 'rollout.mp4'),
                                          note='Native RoboDawn boundary video; not a 1x simulation-time recording.')
            except Exception as exc:
                report.update(status='recording_error', error=str(exc), evaluation_completed=False, success=False)
            report['budget'] = env.control.summary()
            report['timing'] = timing.summary(time.monotonic() - start if start is not None else 0., env)
            report['motion'] = dict(time_dilation_factor=env.motion_speed, planning_calls=env.plan_calls,
                                   last_plan_time_dilation_factor=getattr(env, 'last_plan_speed', None))
        if client is not None:
            report['tokens'] = client.total_usage
            report['model_requests'] = len(client.records)
            report['model_http_attempts'] = sum(len(r['attempts']) for r in client.records)
            report['model_seconds'] = sum(r['wall_s'] for r in client.records)
            report['model_images_including_primer'] = sum(r['image_count'] for r in client.records)
            report['usage_missing_requests'] = sum(not r['usage'] for r in client.records)
        report['process_wall_s'] = time.monotonic()-cold_start
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if env is not None:
            env.close()
    return 0 if report['evaluation_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
