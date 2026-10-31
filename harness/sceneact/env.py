"""SceneAct protocol around RoboDawn's native commands and motion execution."""
from dataclasses import asdict
import os
import sys
import time
from unittest.mock import patch

from harness.core.env import EpisodeInfo
from harness.robotwin.env import RoboTwinDiscreteEnv, ROBOTWIN_ROOT
from .episode import EpisodeControl, EpisodeFinished
from .execution import set_motion_speed, install_step_budget

# Same pinned source checkout used by SceneAct; no package installation needed.
sys.path.insert(0, str(ROBOTWIN_ROOT / 'envs/curobo/src'))


class MatchedEnv(RoboTwinDiscreteEnv):
    strict_execution = True

    def __init__(self, job, reference, render_device=None):
        self.job, self.reference, self.render_device = job, reference, render_device
        super().__init__(job['task'], task_config=job['task_config'], seed=0, instruction_type='seen')
        self.planning_wall_s = self.evaluation_wall_s = self.physics_wall_s = self.video_capture_s = 0.
        self.plan_calls = 0
        self.motion_speed = reference['budget']['motion_speed']

    def reset_episode(self, episode_index=None):
        # Match the frozen SceneAct lifecycle: setup directly with the fixed seed,
        # no online expert replay, seed substitution, or language resampling.
        if self._episode_open:
            self.close_episode()
        job = self.job
        self.episode_index = job['episode_index']
        if hasattr(self, '_official_checker'):
            self.env.check_success = self._official_checker
        import sapien
        import sapien.core

        def create_scene(engine, config):
            sapien.physx.set_scene_config(config)
            return sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(self.render_device)])
        with patch.object(sapien.core.Engine, 'create_scene', create_scene):
            self.env.setup_demo(now_ep_num=self.episode_index, seed=job['seed'], is_test=True, **self.args)
        self._episode_open = True
        if int(self.env.step_lim) != job['step_limit']:
            raise ValueError('official step limit differs from baseline')
        self._create_agent_camera()
        self.env.set_instruction(instruction=job['instruction'])
        self.episode = EpisodeInfo(job['task'], job['task_config'], self.episode_index,
                                   job['seed'], job['instruction'], job['step_limit'])
        self.frames, self.command_log = [], []
        self.control = EpisodeControl(job['step_limit'], self._physics_dt(),
                                     self.reference['budget']['sim_seconds_per_official_step'],
                                     self.reference['budget']['primitive_budget_divisor'])
        if self.env.robot.communication_flag:
            raise ValueError('matched retiming requires in-process cuRobo planners')
        for arm in ('left', 'right'):
            planner = getattr(self.env.robot, arm + '_planner')
            set_motion_speed(planner.motion_gen, self.motion_speed, self._planned)
        install_step_budget(self)
        # Ground truth remains private to the explicitly retained object-fell stop.
        self._initial_objects = {k: v.get('position_cm') for k, v in self.object_states().items()}
        return self.episode

    def _planned(self, seconds, speed):
        self.planning_wall_s += seconds
        self.plan_calls += 1
        self.last_plan_speed = speed

    def _physics_steps(self):
        return self.control.physics_steps

    def _physics_dt(self):
        return float(self.env.scene.get_timestep())

    @property
    def sim_time_s(self):
        return self.control.physics_steps * self.control.physics_dt

    @property
    def steps_used(self):
        return self.control.primitive_executions if hasattr(self, 'control') else 0

    @property
    def step_limit(self):
        return self.control.primitive_limit if hasattr(self, 'control') else self.job['step_limit']

    @property
    def budget_exhausted(self):
        return self.control.status in ('simulation_budget', 'primitive_budget')

    def state(self):
        # Do not expose object truth through memory, prompts or demonstration selection.
        return dict(steps_used=self.steps_used, step_limit=self.step_limit, success=self.success,
                    table_z_cm=self.table_z_cm, left=asdict(self.arm_state('left')),
                    right=asdict(self.arm_state('right')))

    def execute(self, cmd):
        self.control.begin_action()
        before = self.control.physics_steps
        # The native take_action counter is not an evaluation budget in this protocol.
        self.env.take_action_cnt = 0
        start = time.monotonic()
        try:
            try:
                result = super().execute(cmd)
            except EpisodeFinished as exc:
                result = dict(command=cmd.text(), kind=cmd.kind, arm=cmd.arm,
                              ok=exc.status == 'official_success', terminal=exc.status,
                              note=f'episode ended: {exc.status}')
        finally:
            self.control.primitive_executions += int(self.control.physics_steps > before)
        result.update(physics_steps=self.control.physics_steps-before,
                      robot_seconds=(self.control.physics_steps-before)*self.control.physics_dt,
                      steps_used_after=self.steps_used, seconds=time.monotonic()-start,
                      after={arm: asdict(self.arm_state(arm)) for arm in ('left', 'right')})
        return result

    def _record_frame(self, caption=''):
        start = time.monotonic()
        try:
            return super()._record_frame(caption)
        finally:
            self.video_capture_s += time.monotonic()-start

    def devices(self):
        device = self.env.scene.render_system.device
        if device.cuda_id != 0:
            raise ValueError('renderer did not select worker CUDA device 0')
        expected = os.environ.get('EXPECTED_GPU_PCI')
        if expected and device.pci_string.lower().split(':')[-2:] != expected.lower().split(':')[-2:]:
            raise ValueError('renderer PCI device differs from assigned physical GPU')
        return dict(render_pci=device.pci_string, render_cuda=device.cuda_id)

    def views(self, obs, cameras):
        required = [name for camera in cameras
                    for name in (('left_camera', 'right_camera') if camera == 'wrist' else (camera,))]
        missing = [name for name in required if name not in obs['images']]
        if missing:
            raise ValueError(f'native RoboDawn cameras missing from observation: {missing}')
        return super().views(obs, cameras)
