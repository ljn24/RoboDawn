"""Official success, simulation/action budgets and failed-decision diagnostics."""
import math


class OfficialEvaluationError(Exception):
    """An evaluator error must not be treated as a recoverable action failure."""


class EpisodeFinished(Exception):
    """Normal termination, including success reached partway through a primitive."""

    def __init__(self, status):
        super().__init__(status)
        self.status = status


class EpisodeControl:
    def __init__(self, official_steps, physics_dt, sim_seconds_per_official_step=.1,
                 primitive_budget_divisor=10):
        for value in (official_steps, physics_dt, sim_seconds_per_official_step,
                      primitive_budget_divisor):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('episode budget parameters must be finite and positive')
        self.official_steps = official_steps
        self.physics_dt = physics_dt
        self.sim_time_limit_s = official_steps * sim_seconds_per_official_step
        self.physics_step_limit = math.ceil(self.sim_time_limit_s / physics_dt)
        self.primitive_limit = math.ceil(official_steps / primitive_budget_divisor)
        self.physics_steps = self.primitive_attempts = 0
        self.primitive_executions = 0
        self.consecutive_failures = self.total_failures = 0
        self.success = False

    @property
    def status(self):
        if self.success:
            return 'official_success'
        if self.physics_steps >= self.physics_step_limit:
            return 'simulation_budget'
        if self.primitive_executions >= self.primitive_limit:
            return 'primitive_budget'
        return None

    def after_decision(self, result):
        # A partially successful sequence is not a failed decision.
        failed = not result['ok'] and not any(r['result']['ok'] for r in result['executed'])
        self.consecutive_failures = self.consecutive_failures + 1 if failed else 0
        self.total_failures += int(failed)

    def begin_action(self):
        if self.status:
            raise EpisodeFinished(self.status)
        self.primitive_attempts += 1

    def after_step(self, success):
        # Physics is counted at scene.step(), before rendering or evaluation.
        self.success = self.success or bool(success)
        # The final admitted primitive can finish. Its execution count must not
        # interrupt execution; success takes precedence at the time boundary.
        if self.success or self.physics_steps >= self.physics_step_limit:
            raise EpisodeFinished(self.status)

    def summary(self):
        return dict(official_step_limit=self.official_steps,
                    sim_time_limit_s=self.sim_time_limit_s,
                    physics_step_limit=self.physics_step_limit,
                    physics_steps=self.physics_steps,
                    sim_time_s=self.physics_steps*self.physics_dt,
                    primitive_limit=self.primitive_limit,
                    primitive_attempts=self.primitive_attempts,
                    primitive_executions=self.primitive_executions,
                    consecutive_failures=self.consecutive_failures,
                    total_failures=self.total_failures)
