"""Export locally downloaded e0/e1-e4/retry manifests and summaries from ja24.

Usage: python -m harness.sceneact.export_reference --input DIR --output DIR
Input names: {e0,e1-e4,retry}-{manifest,summary}.json and source.tar.gz.
No credentials, raw model inputs, or simulator object poses are exported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import yaml


def export(source, output):
    prefixes = ['e0', 'e1-e4', 'retry']
    manifests = [json.loads((source / f'{p}-manifest.json').read_text(encoding='utf-8')) for p in prefixes]
    first = manifests[0]
    code_hashes = lambda m: {k: v for k, v in m['sources'].items() if k.startswith('sceneact/')}
    for manifest in manifests:
        if manifest['episode_defaults'] != first['episode_defaults'] or code_hashes(manifest) != code_hashes(first):
            raise ValueError('baseline source/defaults mismatch')
    chosen, jobs = {}, {}
    for prefix, manifest in zip(prefixes, manifests):
        for job in manifest['jobs']:
            key = job['task'], job['episode_index']
            data = {k: job[k] for k in ('task', 'mode', 'task_config', 'episode_index', 'seed', 'instruction')}
            if key in jobs and jobs[key] != data:
                raise ValueError(f'conflicting episode: {key}')
            jobs[key] = data
        summary = json.loads((source / f'{prefix}-summary.json').read_text(encoding='utf-8'))
        for result in summary['results']:
            key = result['task'], result['episode_index']
            if key not in chosen and result['evaluation_completed']:
                chosen[key] = result
    if len(jobs) != 250 or set(chosen) != set(jobs) or sum(r['success'] for r in chosen.values()) != 135:
        raise ValueError('reference must contain 135 successful, 250 completed paired episodes')
    with tarfile.open(source / 'source.tar.gz') as archive:
        steps = yaml.safe_load(archive.extractfile('RoboTwin/env_cfg/task_config/_eval_step_limit.yml').read())
    rows = []
    for (task, index), job in sorted(jobs.items(), key=lambda item: (item[0][1], item[0][0])):
        result = chosen[task, index]
        episode = result['episode']
        if (result['seed'], result['instruction']) != (job['seed'], job['instruction']):
            raise ValueError(f'result differs from manifest: {task}/{index}')
        job.update(id=f'{len(rows):03d}_c2r_{task}_e{index}', step_limit=steps[task],
                   baseline=dict(success=result['success'], status=result['status'],
                                 **{k: episode.get(k) for k in ('budget', 'timing', 'decisions', 'image_requests')}))
        rows.append(job)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    defaults = first['episode_defaults']
    reference = dict(
        protocol='sceneact_54pct_v1', baseline_successes=135, baseline_episodes=250,
        instruction_version=first['instruction_version'],
        sources=[dict(host='ja24', batch=m['config']['output'],
                      manifest_sha256=sha(source / f'{p}-manifest.json'),
                      summary_sha256=sha(source / f'{p}-summary.json')) for p, m in zip(prefixes, manifests)],
        source_archive_sha256=sha(source / 'source.tar.gz'), sceneact_source_hashes=code_hashes(first),
        robotwin_input_hashes={k.removeprefix('RoboTwin/'): v for k, v in first['sources'].items() if k.startswith('RoboTwin/')},
        dependencies={k: first['dependencies'][k] for k in ('RoboTwin', 'RoboTwin/envs/curobo')},
        budget={k: defaults[k] for k in ('sim_seconds_per_official_step', 'primitive_budget_divisor', 'motion_speed')},
        process_timeout_s=first['config']['evaluation']['process_timeout_s'],
        max_consecutive_infrastructure_errors=10, model=defaults['model'],
        cameras=defaults['benchmark_profile']['cameras'], jobs=rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'reference.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Verified 135/250 baseline; wrote', output / 'reference.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export(args.input, args.output)
