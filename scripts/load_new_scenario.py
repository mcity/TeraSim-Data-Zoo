from scenparse import ScenarioProcessor, ScenarioVisualizer
from scenparse.utils import read_data
from waymo_open_dataset.protos import scenario_pb2
import copy

waymo_data_list_original = read_data("/home/led/Documents/WOMD/validation/uncompressed_scenario_validation_validation.tfrecord-00000-of-00150")
waymo_data_list = read_data("validation.tfrecord-00000-of-00150-new",)
print(len(waymo_data_list_original))
print(len(waymo_data_list))
for i in range(len(waymo_data_list_original)):
    if waymo_data_list_original[i].scenario_id != "b26ba622ddc9d7d8":
        continue

    if waymo_data_list[i].scenario_id != "b26ba622ddc9d7d8":
        continue

    new_scenario = copy.deepcopy(waymo_data_list_original[i])
    for t in range(91):
        assert len(waymo_data_list[i].dynamic_map_states[t].lane_states) == 36
        new_scenario.dynamic_map_states[t].CopyFrom(waymo_data_list[i].dynamic_map_states[t])

    sv = ScenarioVisualizer(new_scenario)
    sv.save_video(base_dir="")
