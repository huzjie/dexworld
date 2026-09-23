# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""演示：性能剖析。"""
from dexworld.data.sim_engine import TactileSimEngine
from dexworld.model.world_model import DexWorldModel
from dexworld.utils.profiler import Profiler


def main():
    eng = TactileSimEngine(seed=0, vision_size=8)
    trajs = eng.generate_dataset(episodes=4, horizon=6)
    model = DexWorldModel(vision_dim=64, tactile_dim=272, action_dim=7)
    prof = Profiler()
    for traj in trajs:
        prof.start("train_step")
        model.train_step(traj.states[:2], traj.states[2:3])
        prof.stop("train_step")
        prof.start("predict")
        model.predict(traj.states[:2])
        prof.stop("predict")
    print(prof.report())


if __name__ == "__main__":
    main()
