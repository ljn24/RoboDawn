# RoboDawn under the SceneAct 54% baseline protocol

This entry evaluates RoboDawn zero-shot against the **135/250 Adaptive Full**
SceneAct baseline. The original `harness/run_robotwin_eval.py` remains available.
These are adapted protocol results, not a reproduction of RoboDawn's published score.

## Fixed-first one-shot on the selected 20 tasks

Use the new `one-shot-20` preset:

```bash
../SceneAct/.venvs/robotwin/bin/python -B -m harness.sceneact.run_eval \
  --preset one-shot-20 --output results/sceneact54_oneshot20_actions2x --dry-run

../SceneAct/.venvs/robotwin/bin/python -B -u -m harness.sceneact.run_eval \
  --preset one-shot-20 --output results/sceneact54_oneshot20_actions2x \
  --api-key-file ../SceneAct/configs/models/deepseek_flash.key
```

`harness/configs/sceneact/one_shot_20.json` freezes the 20 tasks from SceneAct's
`configs/demos/tasks.yaml`. Each uses the five original C2R episodes (e0–e4),
with unchanged IDs, seeds and reviewed instructions: **100 episodes total**.
The preset defaults to `reference_actions2x.json`, exactly matching the completed
`results/sceneact54_zeroshot_actions2x` run: `ceil(step_lim / 5)` physical primitives,
`step_lim × 0.2` simulation seconds, 0.5 motion speed, 10800 s watchdog,
and the same DeepSeek low-effort request settings. The older zero-shot default
and the protocol table below use `/10`; do not use that default for this experiment.

Every request contains the native command primer plus exactly one RoboDawn expert
demo, loaded directly from `demos/robotwin2/expert/<task>/demo.json` and its images.
No `+2` entry, grasp-side matching, evaluation-state lookup or fallback is used.
Missing/corrupt first demos fail preflight. Native task-demo prompt wording is
restored for this mode. Current observations remain the same four RGB views.
Across these 20 tasks, each request contains 15–48 images including the primer,
task demo and current observations.

The expert bank is hashed and copied into the isolated workspace for one-shot
runs. Each worker saves `demo_task/`, records `shots=1`, `demo_selection=fixed_first`
and `demo_source`, and accounts for all demo images/tokens in model usage.
The legacy metric name `model_images_including_primer` includes task-demo images
as well. Use a fresh output directory; `--resume` only accepts an identical manifest.
`--job-ids` can select a smoke episode within the preset without changing its ID.
Default concurrency is still 8 GPUs × 3 workers; adjust `--gpus` if GPUs are occupied.

Validation: 25 offline protocol/one-shot tests pass, and the 100-episode dry run
passes dependency and demo image checks. The model/budget settings were compared
against the completed actions2x manifest.

The 2026-11-05 run finished with **10/100 successes (10%)**, compared with
**2/100 (2%)** for zero-shot on the exact same selected episodes. There were
99 normally completed evaluations and one `model_api_error` on
`190_c2r_rotate_qrcode_e3` (`'NoneType' object is not subscriptable`). That episode
was intentionally not rerun and remains in the 100-episode denominator.
Normal failures comprised 63 simulation-budget stops, 13 primitive-budget stops
and 13 object-fell stops. Local artifacts are in
`results/sceneact54_oneshot20_actions2x/`; results are ignored by Git.

## Frozen reference

`harness/configs/sceneact/reference.json` was exported from ja24:

- `deepseek_low_50ep_sparse_views_1029`: 50 completed, 29 successful.
- `deepseek_low_200ep_sparse_views_1029`: 179 completed, 101 successful.
- `deepseek_low_200ep_sparse_views_retry1_21workers_1030`: 21 completed, 5 successful.

The three manifests have identical SceneAct source hashes and episode defaults.
The export keeps the first normally completed result per task/episode index;
retries fill infrastructure failures only. It stores all 250 actual seeds, exact
reviewed/seen instructions, official limits, baseline results/statistics, source
hashes, model parameters and cameras. Manifest/summary/archive SHA-256 hashes
identify the source evidence. `export_reference.py` rebuilds it from downloaded
files without copying credentials or privileged object state.

`episode.py` and `timing.py` are copied unchanged from this baseline's source
archive; their hashes can be compared with `sceneact_source_hashes` in the reference.

## Protocol

| Setting | Evaluation value |
|---|---|
| Tasks/scenes | 50 tasks × 5 C2R scenes, exact paired seeds and instructions |
| Scene configuration | Native RoboDawn `demo_randomized` |
| Model | deepseek-flash, Chat Completions, explicit `reasoning_effort=low` |
| Request defaults | Native `temperature=0.0`, `max_tokens=8000`, DeepSeek `thinking.type=enabled`; no output schema |
| Model request limits | 300 s per attempt, 900 s retry window, 2 retries |
| Simulation budget | official `step_lim × 0.2` seconds |
| Execution budget | `ceil(step_lim / 10)` primitives that advance physics |
| Physics | native 250 Hz; includes gripper settling and waits |
| Motion speed | cuRobo `MotionGenPlanConfig.time_dilation_factor=0.5`, both arms |
| Process watchdog | 10800 s, process group terminated on timeout |
| Parallelism | physical GPUs 0–7, 3 workers each, 24 total |
| Current observations | Native RoboDawn RGB: overview, top-down, left wrist and right wrist |
| Demonstrations | command primer only; no task demonstration |
| Additional early stop | native `object_fell`, retained intentionally |

The final admitted primitive may finish; simulation time can interrupt it.
Success takes precedence at the time boundary. Partial physical executions
count once; nonexecuted planning failures and invalid decisions do not consume
execution budget. Attempts remain separately logged. If the native executor
advances physics while a plan fails, that command does count as an execution.

Native 45-turn, official action-counter, consecutive-parse-failure and repeated
`done` limits are not used by this entry. Invalid decisions and unsuccessful
`done` replies continue until the shared budgets/watchdog or object-fell stop.
API failure after the bounded retries is an infrastructure failure, not a new
round of policy-level retries. Ten consecutive infrastructure results latch a
dispatch stop; running workers finish normally.

Model requests reuse the native `ChatClient._body()` implementation. The bounded
transport retry/error policy above remains from the comparison harness, rather
than reverting to native retries. The explicit low-effort override is retained
from the baseline, expressed as Chat Completions `reasoning_effort=low`; it is not
the original RoboDawn default for DeepSeek. No automatic parameter relaxation or
endpoint fallback is applied. The new endpoint's acceptance of multimodal input,
temperature, token cap and reasoning parameters still requires a live smoke test.
These are requested parameters; offline validation cannot establish whether the
provider honors every parameter.

`reference.json` remains the immutable SceneAct evidence (Responses, no explicit
temperature/token cap, JSON Schema). The batch manifest's `baseline_model` keeps
those settings, while `model` and each worker's `model_config` record the actual
RoboDawn settings. The methods no longer have identical API/sampling settings.

Prompts reuse `system_prompt()` and the unchanged `turn_text()`, with no appended
budget instructions, and include only the original command primer. The upstream
repository does not ship a standalone runnable published zero-shot preset.
The system's demonstration paragraph and the primer introduction, closing text
and acknowledgement explicitly describe command examples without a task
demonstration. Other control instructions and primer images/commands are retained;
the default wording for runs with task demonstrations is unchanged.
Runtime instructions/state still come from the paired evaluation, including the
changed meaning of the existing `steps used` field. Parsing and feedback follow
the native loop: execute valid commands even when others are invalid, append the
individual errors, accept an empty command list, and retain original parse-error
and `done` feedback. A malformed JSON reply is retried on the next policy turn;
the native parse-failure and `done` termination limits remain disabled.

Native RoboDawn commands, memory, grid/marker annotations and feedback remain.
Camera selection uses the native `AgentConfig` default: `agent_camera`,
`top_camera`, `wrist` (expanded to `left_camera` and `right_camera`). The native
camera placement, including the overview/top-down table-height offset, benchmark
profile and recorded primer are retained. SceneAct's side cameras are not added.
The reference's `cameras` field records the frozen SceneAct baseline only; it does
not configure RoboDawn's observations. The manifest records
`observation_mode=robodawn_native`, and each worker saves its camera selection in
`agent_config.json`. Camera views/extrinsics are intentionally not matched between
methods. Missing required native views fail the episode instead of silently
reducing its image inputs. Object poses are read only by the
retained private object-fell checker, not sent in prompts or used to select demos.
The SceneAct reset lifecycle is used: no online expert replay or seed replacement.

The motion executor remains native RoboDawn, with its cuRobo retiming aligned.
This does **not** make its collision models, planner settings or gripper settling
identical to SceneAct. SceneAct also uses RGB-D, SAM and GraspGenX, while this
baseline uses RoboDawn's RGB interface. These are system-level comparisons.

## Run

From the RoboDawn repository, using the existing SceneAct environment and assets:

```bash
../SceneAct/.venvs/robotwin/bin/python -B -m harness.sceneact.run_eval \
  --output results/sceneact54_zeroshot --dry-run
```

The dry run checks pinned RoboTwin/cuRobo commits, tracked modifications and
frozen benchmark configuration/description hashes. It imports no simulator and
sends no model request. Large scene assets and all installed packages are not
individually fingerprinted; the same prepared environment/assets are required.

For the full **8 GPUs × 3 workers** run:

```bash
../SceneAct/.venvs/robotwin/bin/python -B -u -m harness.sceneact.run_eval \
  --output results/sceneact54_zeroshot \
  --api-key-file ../SceneAct/configs/models/deepseek_flash.key
```

Alternatively set `DEEPSEEK_API_KEY`. The key is passed through the worker
environment and never written to the manifest. RoboDawn needs no SAM/GraspGenX
services, so GPU 0 also runs three workers. The launcher binds both CUDA and the
renderer to the assigned physical GPU and verifies the renderer PCI identity.

Use `--robotwin-root PATH` and `--python PATH` for another checkout/environment.
For a small run use `--gpus 0 --workers-per-gpu 1 --job-ids ID[,ID]` and a fresh
output directory. Job IDs come from the reference, e.g.
`029_c2r_place_empty_cup_e0`; indices are preserved when filtering.

`--resume` requires identical sources, reference, configuration and dependencies.
It skips terminal records, including failures. Interrupted attempts remain
incomplete. Explicit infrastructure reruns use `--job-ids` in a new output;
merge only first normally completed outcomes, never choose best outcomes.

## Statistics

Each worker writes `result.json`, `model.jsonl`, complete `model_inputs/`,
`trace.jsonl`, current images, primer copy, memory and a boundary video.
The batch freezes harness/primer source in `workspace/`, and writes
`manifest.json`, `reference.json`, `devices.json`, `summary.json`, `summary.csv`.

- `decisions`: RoboDawn decision rounds; `model_requests` and
  `model_http_attempts`: logical requests and actual transport attempts.
- `image_views`: logical current-image requests, including first;
  `image_requests`: subsequent views, matching SceneAct's field convention.
  `current_images` counts individual current-view images;
  `model_images_including_primer` also includes repeated primer images.
- `budget`: attempts, physical executions, physics steps, simulation seconds
  and all limits; these are not interchangeable “steps.”
- `tokens`: provider input/output/total, reasoning and cached token counts
  when available; `usage_missing_requests` flags unavailable usage. Reasoning
  tokens are a subset of output, not added to it again.
- `timing`: task wall time, system inference, observation, VLM decision,
  planning, execution, evaluator and logging/video time. cuRobo planning and
  logging are subtracted from execution before aggregation. There is no separate
  VLM initialization in RoboDawn. Cold start/reset/process time are separate.

Summary averages are provided for successful, completed and all-returned episodes,
with per-field sample counts. Use **successful** averages for comparison with
SceneAct's experiment table. Missing fields remain null. Success-rate denominator
is the full scheduled set, including not-yet-run and infrastructure failures.
Native boundary video is not a continuous 1x-time recording like SceneAct's video.

## Validation

```bash
../SceneAct/.venvs/robotwin/bin/python -B -m unittest discover \
  -s tests -p 'test_sceneact*.py' -q
```

Offline tests do not make network requests or start a simulator. The existing real
simulator and model smoke artifacts below were produced during the earlier
three-view/Responses adaptation, before restoring native RoboDawn cameras and
request/prompt behavior. They do not validate the current four-view/Chat
Completions path; that smoke run is pending:

- Three current views at 640×480; move, open-gripper and wait executed with
  3 executions / 292 physics steps, including gripper settling; both-arm cuRobo
  calls received `time_dilation_factor=0.5`.
- A separate boundary smoke test limited the next command to three remaining
  physics steps: it stopped after exactly three, reported `simulation_budget`,
  and counted the partially executed command once. This artificial boundary
  override applies only to that smoke artifact, not the evaluation reference.
- A real DeepSeek first-round request returned valid commands in 10.25 s:
  4975 input / 1515 output tokens, including 1314 reasoning tokens. Its first
  command executed for 796 physics steps with the matched speed.

The 20 offline tests cover pairing, exact frozen budget code, per-tick evaluation,
speed injection, removed native limits, retries, UTF-8 logging, timeout cleanup,
and statistics. Agent checks use native camera selection and count four current
images per decision. A differential test compares complete model messages against
the native loop supplied with only the primer (allowing the explicit zero-shot
system paragraph), through malformed JSON,
empty commands, invalid commands, `done`, and mixed valid/invalid commands. Request
bodies are compared against the native client with an explicit low-effort override.

Local smoke artifacts are in `results/sceneact54_smoke/`; these are checks, not
benchmark episodes or success-rate measurements. The 250-episode evaluation
has not been launched. The original full test suite has four RoboDojo lifecycle
failures (`KeyError: base_url`), also reproduced in a clean upstream checkout;
those unrelated files were not changed.
