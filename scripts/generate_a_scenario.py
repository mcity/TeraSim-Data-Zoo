from scenparse import ScenarioProcessor, ScenarioVisualizer
from scenparse.utils import read_data
from pathlib import Path
import re
import multiprocessing
import signal



def generate_a_test_scenario(scenario):
    sp = ScenarioProcessor(scenario, sumonize_config={"custom_lanewidth": True})
    sp.generate_sumonic_tls(
        # veh_states_file=f"/home/led/Documents/maps/waymo-testing/{scenario.scenario_id}/{scenario.scenario_id}-sumonicassign.json",
        generator_config={"yellow_duration":0, "delta_t": 4})
    sp.save_SUMO_netfile(base_dir=f"/home/led/Documents/maps/waymo-testing/{scenario.scenario_id}", verbose=True)
    


fails = [(69, '3308cbd417840ae3'),
(56, '9465fb15b456855a'),
(138, '981f5c4f61505759'),
(76, 'ff6686b0e98d66ae'),
(149, 'ab7571715a1d5193'),
(45, '8738e8c0200056fe'),
(141, '38e2986c8098692f'),
(124, '8712a83f2e267651'),
(36, 'a869787ec83d5c8a'),
(63, '4c99903f949e8100'),
(83, '244d7ba1c6513b17'),
(91, '9036b3f956b09fc0')]

emptys = [
    (56, "9465fb15b456855a"),
(138, "981f5c4f61505759"),
(76, "ff6686b0e98d66ae"),
(149, "ab7571715a1d5193"),
(45, "8738e8c0200056fe"),
(141, "38e2986c8098692f"),
(36, "a869787ec83d5c8a"),
(63, "4c99903f949e8100"),
(91, "9036b3f956b09fc0"),
]

# for i, id in emptys:
#     waymo_data_list = read_data("/home/led/Documents/WOMD/testing/testing.tfrecord-{:05d}-of-00150".format(i), [id])
#     try:
#         generate_a_test_scenario(waymo_data_list[0])
#     except Exception as e:
#         print(e)
