"""Testable cuRobo retiming and per-physics-step evaluation hooks."""
import time

from .episode import EpisodeFinished, OfficialEvaluationError


def set_motion_speed(motion_gen, speed, on_plan):
    """Set cuRobo's actual retiming option, without resampling trajectories by hand."""
    original = motion_gen.plan_single

    def plan(start, goal, plan_config, *args, **kwargs):
        plan_config.time_dilation_factor = speed
        begin = time.monotonic()
        try:
            return original(start, goal, plan_config, *args, **kwargs)
        finally:
            on_plan(time.monotonic() - begin, speed)
    motion_gen.plan_single = plan


def install_step_budget(adapter):
    task = adapter.env
    original_step = task.scene.step
    adapter._official_checker = task.check_success

    def step():
        control = adapter.control
        if control.success or control.physics_steps >= control.physics_step_limit:
            raise EpisodeFinished(control.status)
        start = time.monotonic()
        original_step()
        adapter.physics_wall_s += time.monotonic() - start
        control.physics_steps += 1
        task._update_render()
        start = time.monotonic()
        try:
            task.eval_success = task.eval_success or bool(adapter._official_checker())
        except Exception as exc:
            raise OfficialEvaluationError(f'official success evaluation failed: {exc}') from exc
        finally:
            adapter.evaluation_wall_s += time.monotonic() - start
        control.after_step(task.eval_success)
    task.scene.step = step
    # Native take_action/settling ask again: do not repeat stateful checks.
    task.check_success = lambda: adapter.control.success
