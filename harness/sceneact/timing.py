"""Exclusive wall-clock stages; simulator time is reported separately."""
from contextlib import contextmanager
import time


class Timings:
    def __init__(self):
        self.seconds, self.calls = {}, {}
        self.details = {}

    @contextmanager
    def detail(self, name):
        """Diagnostic spans may overlap; do not add them to exclusive totals."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.details[name] = self.details.get(name, 0.) + time.monotonic()-start

    @contextmanager
    def measure(self, name):
        start = time.monotonic()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.) + time.monotonic()-start
            self.calls[name] = self.calls.get(name, 0) + 1

    def summary(self, wall_s, env):
        inference = sum(v for k, v in self.seconds.items() if k not in ('execution', 'logging', 'evaluation'))
        sim = env.sim_time_s
        return dict(task_wall_s=wall_s, system_inference_s=inference, motion_sim_s=sim,
                    inference_over_motion=inference/sim if sim else None,
                    physics_wall_s=env.physics_wall_s,
                    component_wall_s=self.seconds, calls=self.calls,
                    detail_wall_s=self.details,
                    unaccounted_wall_s=wall_s-sum(self.seconds.values()))
