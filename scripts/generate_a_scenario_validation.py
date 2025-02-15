from scenparse import ScenarioProcessor, ScenarioVisualizer
from scenparse.utils import read_data
from pathlib import Path
import re
import multiprocessing
import signal

from scenparse import ScenarioProcessor, ScenarioVisualizer
from scenparse.utils import read_data
from pathlib import Path
import re
import multiprocessing
import signal

# 定义超时后的处理函数
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# 使用上下文管理器进行超时控制
def run_with_timeout(func, args=(), timeout=60):
    # 注册信号处理程序
    signal.signal(signal.SIGALRM, timeout_handler)
    
    # 设置超时时间
    signal.alarm(timeout)
    
    try:
        # 运行你的函数
        result = func(*args)
    finally:
        # 取消定时器（避免不必要的超时信号）
        signal.alarm(0)
    
    return result

def generate_a_test_scenario(scenario):
    sp = ScenarioProcessor(scenario, sumonize_config={"custom_lanewidth": True})
    # sp.generate_sumonic_tls(
    #     # veh_states_file=f"/home/led/Documents/maps/waymo-testing/{scenario.scenario_id}/{scenario.scenario_id}-sumonicassign.json",
    #     )
    sp.save_SUMO_netfile(base_dir=f"/home/led/Documents/maps/waymo-validation/{scenario.scenario_id}", verbose=True)
    
data = [
    (13, "dd001b11d38703d2"),
    (67, "328139a82e7ecf9f"),
    (17, "80e53e1762d71719"),
    (82, "de92263018ee4f23"),
    (94, "30dbe113fa913c62"),
    (143, "f20e7077dcf0ffe4"),
    (78, "6b37c52784b41589"),
    (132, "ddc911c3a783e188"),
    (97, "b19533b636582ec9")
]



with open("fail2.log", "w") as f:

    for i, id in data:
        waymo_data_list = read_data("/media/led/WD_2TB/WOMD/validation/validation.tfrecord-{:05d}-of-00150".format(i), [id])
        try:
            run_with_timeout(generate_a_test_scenario, args=(waymo_data_list[0],), timeout=60)
        except Exception as e:
            f.write(f"#### {i} | {id} ####\n")
            print(e)
            f.flush()
