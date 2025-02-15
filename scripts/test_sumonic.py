from scenparse import ScenarioProcessor, ScenarioVisualizer
from scenparse.utils import read_data
from pathlib import Path
import re

def extract_lane_data(warning_string: str):
    """
    This function takes a string containing multiple warnings, 
    extracts the relevant lane data and returns a list of tuples
    in the format (Lane1_Num, Lane1_Sub, Lane2_Num, Lane2_Sub).
    """
    # Regular expression to match the desired pattern
    pattern = re.compile(r"lane '(\d+)_(\d+)' and lane '(\d+)_(\d+)'")

    # Find all matches
    matches = pattern.findall(warning_string)

    # Convert matches to list of tuples in the required format
    result = []
    for match in matches:
          result.extend([(int(match[0]),), (int(match[2]),)])
    return result


ped = ['e9653159c09f5c72', 'a067a3788873f5be', 'ae92386e0686d4d6', '914f16ffef6529c5', '4f7aa8ac2ff60093', 'e3d1b5e3c9e05d41', '5d6f6adb8f60a8e', 'a2fd6176c7695739', 'ee5e5fe91cd6c6e7', '3b523aceb232aa6f', '1870b2208fe4ade9', '9646b687f1411768', 'b6c121ee6748be2e', 'a8c2bfa46d9e6c4', '7fcb65089a2bb672', '1e41e4f90a5cae0a', 'a200040e65ba224c', 'c92f5e81941151a4', '57e7012e0a98e3ec', '371a64917512b4c9', '3a7c04d2bd7807eb', 'bb2c545bd3133e12', '7e0a3b06b62db8bf', '2cdef5a38212b115', 'e5cb9f7995a95a9b', 'c496f211b25122b1', '5420f153f073b690', 'f189ad0f80bfcbca', 'd4033db5f9f9b088', '2c5702a36704c1ba', '5567660e03ce1d5d', 'd9b4c97449084110', '26496133a911e90e', '128d8b9642c6f6cd', 'e9b95352cc566fa2', '14eeba302e55cc22', '5278b05f63b435d3', '99d7da4bd15d811b']

waymo_data_list = read_data("/media/led/WD_2TB/WOMD/validation/validation.tfrecord-00000-of-00150",  )
for i, scenario in enumerate(waymo_data_list):
    sv0 = ScenarioVisualizer(scenario, types_to_draw=[0,1,2,4,5], )
    sv0.save_map(base_dir="outputs/imgs-maps")
    # sv0.save_video(base_dir=".",)

    # try: 5d6f6adb8f60a8e 140117c81b4ef9fb a2fd6176c7695739 ee5e5fe91cd6c6e7 e9653159c09f5c72
    # print(f"////////{i} of {len(waymo_data_list)}//////")
    # sp = ScenarioProcessor(scenario, sumonize_config={"custom_lanewidth": True, "add_sidewalk": True})
    # # sp.plot_waymonic_map(base_dir=f"/home/led/Documents/maps/waymo/{scenario.scenario_id}")
    # sp.generate_sumonic_tls(
    #         veh_states_file=f"/home/led/Documents/maps/waymo/{scenario.scenario_id}/{scenario.scenario_id}-sumonicassign.json",
    #         generator_config={"yellow_duration":30})
    # # sp.plot_sumonic_map(base_dir=f"/home/led/Documents/maps/waymo/{scenario.scenario_id}",)
    # sp.save_SUMO_netfile(base_dir=f"/home/led/Documents/maps/{scenario.scenario_id}", verbose=True)
        
        

        
    #     intersections, dynamic_states, dict_data = sp.generate_waymonic_tls(veh_states_file=f"map/{sp.scenario.scenario_id}/{sp.scenario.scenario_id}-assignment.json", return_data="all", save_file=True)
    # except:
    #     continue