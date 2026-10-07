"""Run LeRobot's LIBERO evaluation unchanged and log every environment reset.

Each LiberoEnv.reset call appends one JSON line to RESET_LOG: task, sub-environment index,
initial-state index and SHA-1 hash, and whether the call was an explicit reset (an episode
that is counted) or an autoreset from inside the vector env's step (an episode that is
discarded). The hash uses the same function as the OpenVLA wrapper, so episodes can be
paired across models by initial state.

Usage: RESET_LOG=/path/resets.jsonl python run_lerobot_libero_logged.py <lerobot-eval arguments>
"""
import hashlib
import inspect
import json
import os
import sys

import numpy as np

import lerobot.envs.libero as libero_env

LOG = open(os.environ["RESET_LOG"], "a")
_original_reset = libero_env.LiberoEnv.reset


def state_hash(state):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(state, dtype=np.float64)).tobytes()).hexdigest()


def reset(self, seed=None, **kwargs):
    states = self._init_states if self.init_states else None
    idx = self.init_state_id % len(states) if states is not None else None
    caller = "autoreset" if any(f.function == "step" for f in inspect.stack()[1:8]) else "explicit"
    out = _original_reset(self, seed=seed, **kwargs)
    LOG.write(json.dumps({
        "task_id": self.task_id,
        "sub_env": self.episode_index,
        "init_state_idx": idx,
        "init_state_sha1": state_hash(states[idx]) if states is not None else None,
        "caller": caller,
        "seed": seed,
    }) + "\n")
    LOG.flush()
    return out


libero_env.LiberoEnv.reset = reset

from lerobot.scripts.lerobot_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
