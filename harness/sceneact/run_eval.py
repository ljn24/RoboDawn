"""Frozen SceneAct baseline pairing, isolated workers and 8 x 3 scheduling.

Dry-run imports neither the simulator nor torch and makes no model requests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from .client import evaluation_model_config


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'harness/configs/sceneact/reference.json'
ACTIVE_PROCESSES = {}
PROCESS_LOCK = threading.Lock()
STOP_REQUESTED = threading.Event()


def stop_workers():
    STOP_REQUESTED.set()
    with PROCESS_LOCK:
        for process in ACTIVE_PROCESSES.values():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root):
    return {str(p.relative_to(root)): file_hash(p)
            for sub in ('harness', 'demos/robotwin2/primer') for p in sorted((root / sub).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def validate_dependencies(root, reference):
    identities = {}
    for key, relative in [('RoboTwin', ''), ('RoboTwin/envs/curobo', 'envs/curobo')]:
        path = root / relative
        commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != reference['dependencies'][key]['commit']:
            raise ValueError(f'{key} commit mismatch: {commit}')
        # Do not modify or reset a shared checkout. Refuse mismatching tracked code.
        diff = subprocess.check_output(['git', '-C', str(path), 'diff', 'HEAD', '--binary'])
        if diff:
            raise ValueError(f'{key} has tracked modifications')
        identities[key] = dict(path=str(path), commit=commit)
    for relative, expected in reference['robotwin_input_hashes'].items():
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'RoboTwin frozen input differs: {relative}')
    return identities


def select_jobs(reference, ids):
    jobs = [{k: v for k, v in j.items() if k != 'baseline'} for j in reference['jobs']]
    if ids is None:
        return jobs
    requested = ids.split(',')
    if len(set(requested)) != len(requested) or set(requested) - {j['id'] for j in jobs}:
        raise ValueError('duplicate or unknown job IDs')
    return [j for j in jobs if j['id'] in requested]


def metrics(result):
    timing = result.get('timing', {})
    components = timing.get('component_wall_s', {})
    budget = result.get('budget', {})
    return dict(decisions=result.get('decisions'), image_views=result.get('image_views'),
                image_requests=result.get('image_requests'), current_images=result.get('current_images'),
                model_requests=result.get('model_requests'), model_http_attempts=result.get('model_http_attempts'),
                usage_missing_requests=result.get('usage_missing_requests'),
                model_images_including_primer=result.get('model_images_including_primer'),
                model_seconds=result.get('model_seconds'),
                **{k: result.get('tokens', {}).get(k) for k in
                   ('input_tokens', 'output_tokens', 'total_tokens', 'reasoning_tokens', 'cached_tokens')},
                primitive_attempts=budget.get('primitive_attempts'), primitive_executions=budget.get('primitive_executions'),
                physics_steps=budget.get('physics_steps'), motion_sim_s=budget.get('sim_time_s'),
                task_wall_s=timing.get('task_wall_s'), system_inference_s=timing.get('system_inference_s'),
                inference_and_execution_s=(timing['system_inference_s'] + components.get('execution', 0.))
                    if 'system_inference_s' in timing else None,
                **{f'{k}_s': components.get(k) for k in ('observation', 'llm_decision', 'planning', 'execution', 'logging', 'evaluation')})


def aggregate(output, jobs, reference, results):
    baseline = {j['id']: j['baseline'] for j in reference['jobs']}
    selected = [results[j['id']] for j in jobs if j['id'] in results]
    successes = [r for r in selected if r.get('evaluation_completed') and r.get('success')]
    completed = [r for r in selected if r.get('evaluation_completed')]
    averages = {}
    for name, group in [('successful', successes), ('completed', completed), ('all_returned', selected)]:
        values = [metrics(r) for r in group]
        averages[name] = {key: dict(mean=sum(xs)/len(xs) if xs else None, n=len(xs))
            for key in metrics({}) for xs in [[v[key] for v in values if v[key] is not None]]}
    statuses = {}
    for r in selected:
        statuses[r['status']] = statuses.get(r['status'], 0) + 1
    summary = dict(protocol=reference['protocol'], jobs=len(jobs), returned=len(selected),
                   evaluations_completed=len(completed), successes=len(successes),
                   success_rate=len(successes)/len(jobs), statuses=statuses, averages=averages,
                   baseline_successes=sum(baseline[j['id']]['success'] for j in jobs),
                   results=selected,
                   notes=['Success-rate denominator is all scheduled jobs; infrastructure errors remain visible.',
                          'Successful averages match SceneAct reporting; missing statistics are null, not zero.',
                          'image_views includes first current observation; image_requests excludes the first.',
                          'Image counts include logical requests, not HTTP retries. Primer images counted separately.',
                          'RoboDawn uses its native overview, top-down and two wrist views; cameras differ from SceneAct.',
                          'RoboDawn uses Chat Completions, temperature=0 and max_tokens=8000; baseline request settings differ.',
                          'Object-fell early termination retained; native motion planner retained with 0.5 retiming.'])
    write_json(output / 'summary.json', summary)
    rows = [dict(id=r['id'], task=r['task'], seed=r['seed'], success=r['success'],
                 evaluation_completed=r['evaluation_completed'], status=r['status'],
                 baseline_success=baseline[r['id']]['success'], **metrics(r)) for r in selected]
    if rows:
        with (output / 'summary.csv').open('w', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def gpu_devices(gpus):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,pci.bus_id,uuid', '--format=csv,noheader'], text=True)
    devices = {}
    for row in csv.reader(raw.splitlines()):
        index, pci, uuid = [v.strip() for v in row]
        devices[int(index)] = dict(pci=pci.lower(), uuid=uuid)
    if set(gpus) - devices.keys():
        raise ValueError('requested physical GPU is unavailable')
    return devices


def run_job(job, slot, plan, output, devices):
    begin = time.monotonic()
    directory = output / 'jobs' / job['id']
    try:
        return _run_job(job, slot, plan, output, devices)
    except Exception as exc:
        result = dict(job, success=False, evaluation_completed=False, status='infrastructure_error',
                      error=f'{type(exc).__name__}: {exc}', physical_gpu=slot[0], worker_slot=slot[1],
                      process_wall_s=time.monotonic()-begin)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'result.json', result)
        return result
    finally:
        temporary = directory / 'tmp'
        if temporary.exists():
            shutil.rmtree(temporary)


def _run_job(job, slot, plan, output, devices):
    gpu, slot_index = slot
    directory = output / 'jobs' / job['id']
    directory.mkdir(parents=True)
    write_json(directory / 'job.json', job)
    write_json(directory / 'attempt.json', dict(start_unix_s=time.time(), physical_gpu=gpu, worker_slot=slot_index))
    temporary = directory / 'tmp'
    temporary.mkdir()
    env = dict(os.environ, ROBOTWIN_ROOT=plan['robotwin_root'], CUDA_VISIBLE_DEVICES=devices[gpu]['uuid'],
               EXPECTED_GPU_PCI=devices[gpu]['pci'], PYTHONUNBUFFERED='1', PYTHONUTF8='1',
               OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(temporary),
               PATH=str(Path(plan['python']).parent) + ':' + os.environ['PATH'])
    cmd = [plan['python'], '-B', '-m', 'harness.sceneact.worker', '--job', str(directory / 'job.json'),
           '--reference', str(output / 'reference.json'), '--output', str(directory), '--render-device', 'cuda:0']
    begin = time.monotonic()
    timed_out = False
    with (directory / 'worker.log').open('w', encoding='utf-8') as log:
        with PROCESS_LOCK:
            if STOP_REQUESTED.is_set():
                result = dict(job, success=False, evaluation_completed=False, status='interrupted')
                write_json(directory / 'result.json', result)
                return result
            process = subprocess.Popen(cmd, cwd=output / 'workspace', env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            ACTIVE_PROCESSES[process.pid] = process
        try:
            code = process.wait(timeout=plan['process_timeout_s'])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = -signal.SIGTERM
        finally:
            # Reap descendants even if the worker leader exited first.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            with PROCESS_LOCK:
                ACTIVE_PROCESSES.pop(process.pid, None)
    path = directory / 'result.json'
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        if code != 0 and result.get('evaluation_completed'):
            result.update(status='worker_error', evaluation_completed=False, success=False)
    else:
        result = dict(job, success=False, evaluation_completed=False,
                      status='worker_error')
    if timed_out or STOP_REQUESTED.is_set():
        result.update(status='process_timeout' if timed_out else 'interrupted',
                      success=False, evaluation_completed=False)
    result.update(physical_gpu=gpu, worker_slot=slot_index, returncode=code,
                  process_wall_s=time.monotonic()-begin)
    write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--robotwin-root', type=Path, default=ROOT.parent / 'SceneAct/RoboTwin')
    parser.add_argument('--python', type=Path, default=ROOT.parent / 'SceneAct/.venvs/robotwin/bin/python')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--workers-per-gpu', type=int, default=3)
    parser.add_argument('--launch-stagger-s', type=float, default=5.)
    parser.add_argument('--job-ids', help='comma-separated full reference IDs; use a fresh output for subset reruns')
    parser.add_argument('--api-key-file', type=Path, help='optional local key file, never archived')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    reference_path = args.reference.resolve()
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    jobs = select_jobs(reference, args.job_ids)
    gpus = [int(g) for g in args.gpus.split(',')]
    if args.workers_per_gpu < 1 or len(set(gpus)) != len(gpus) or args.launch_stagger_s < 0:
        raise ValueError('invalid worker configuration')
    robotwin_root, output = args.robotwin_root.resolve(), args.output.resolve()
    identities = validate_dependencies(robotwin_root, reference)
    plan = dict(protocol=reference['protocol'], jobs=jobs, reference_sha256=file_hash(reference_path),
                sources=source_hashes(ROOT), robotwin_root=str(robotwin_root), dependencies=identities,
                python=str(args.python.absolute()), gpus=gpus, workers_per_gpu=args.workers_per_gpu,
                launch_stagger_s=args.launch_stagger_s, process_timeout_s=reference['process_timeout_s'],
                budget=reference['budget'], model=evaluation_model_config(reference['model']),
                baseline_model=reference['model'], shots=0, retain_object_fell=True,
                observation_mode='robodawn_native')
    if args.dry_run:
        print(json.dumps({k: v for k, v in plan.items() if k not in ('sources', 'jobs')}, indent=2))
        print(f'{len(jobs)} paired episodes; {len(gpus)*args.workers_per_gpu} worker slots; no requests launched')
        return
    if args.api_key_file:
        os.environ[reference['model']['api_key_env']] = args.api_key_file.read_text(encoding='utf-8').strip()
    if not os.environ.get(reference['model']['api_key_env'], '').strip():
        raise ValueError('set the baseline API-key environment variable or pass --api-key-file')
    devices = gpu_devices(gpus)
    if args.resume:
        if json.loads((output / 'manifest.json').read_text(encoding='utf-8')) != plan:
            raise ValueError('resume manifest differs; use a new output')
        if source_hashes(output / 'workspace') != plan['sources']:
            raise ValueError('frozen source changed')
        if file_hash(output / 'reference.json') != plan['reference_sha256']:
            raise ValueError('frozen reference changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'manifest.json', plan)
        shutil.copy2(reference_path, output / 'reference.json')
        for sub in ('harness', 'demos/robotwin2/primer'):
            shutil.copytree(ROOT / sub, output / 'workspace' / sub, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    write_json(output / 'devices.json', devices)
    results = {}
    for job in jobs:
        path = output / 'jobs' / job['id']
        if (path / 'result.json').exists():
            results[job['id']] = json.loads((path / 'result.json').read_text(encoding='utf-8'))
        elif path.exists():
            result = dict(job, success=False, evaluation_completed=False, status='interrupted')
            write_json(path / 'result.json', result)
            results[job['id']] = result
    pending = iter(j for j in jobs if j['id'] not in results)
    slots = [(gpu, i) for i in range(args.workers_per_gpu) for gpu in gpus]
    consecutive_errors, futures, dispatch_stopped = 0, {}, False
    aggregate(output, jobs, reference, results)
    STOP_REQUESTED.clear()

    def interrupted(signum, frame):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        stop_workers()
        write_json(output / 'interrupted.json', dict(signal=signum, time_unix_s=time.time()))
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        def launch(slot):
            if STOP_REQUESTED.is_set():
                return
            job = next(pending, None)
            if job is None:
                return
            futures[pool.submit(run_job, job, slot, plan, output, devices)] = slot
            if args.launch_stagger_s:
                time.sleep(args.launch_stagger_s)
        for slot in slots:
            launch(slot)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                slot = futures.pop(future)
                try:
                    result = future.result()
                except BaseException:
                    stop_workers()
                    raise
                results[result['id']] = result
                consecutive_errors = 0 if result['evaluation_completed'] else consecutive_errors + 1
                dispatch_stopped = dispatch_stopped or consecutive_errors >= reference['max_consecutive_infrastructure_errors']
                aggregate(output, jobs, reference, results)
                print(result['id'], result['status'], flush=True)
                if not dispatch_stopped:
                    launch(slot)
    summary = aggregate(output, jobs, reference, results)
    print(json.dumps({k: summary[k] for k in ('jobs', 'returned', 'evaluations_completed', 'successes', 'success_rate')}))


if __name__ == '__main__':
    main()
