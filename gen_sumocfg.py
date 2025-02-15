import os
from scenparse.utils.gen_sumocfg import gen_sumocfg

root_dir = "/home/led/Documents/maps"
all_subdirs = sorted(os.listdir(root_dir))
for subdir in all_subdirs:
    gen_sumocfg(root_dir, subdir, )