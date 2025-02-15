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
    sp.generate_sumonic_tls(
        veh_states_file=f"/home/led/Documents/maps/waymo-validation/{scenario.scenario_id}/{scenario.scenario_id}-sumonicassign.json",)
    sp.save_SUMO_netfile(base_dir=f"/home/led/Documents/maps/waymo-validation/{scenario.scenario_id}", verbose=True)


def generate_a_file(file_number: int):
    try:
        waymo_data_list = read_data("/media/led/WD_2TB/WOMD/validation/validation.tfrecord-{:05d}-of-00150".format(file_number),)
    except:
        return
    output_file = "outputs/mapconvert_logs_validation/0927-{:05d}.txt".format(file_number)
    
    with open(output_file, 'w') as file:
        # main loop
        for i, scenario in enumerate(waymo_data_list):
            try:
                print(f"////////[{file_number}] - {i} of {len(waymo_data_list)}//////")
                run_with_timeout(generate_a_test_scenario, args=(scenario,), timeout=90)  # 设置超时为60秒
                
            # error: jump this scenario
            except TimeoutError as te:
                file.write(f"{scenario.scenario_id} fails due to timeout. Jump this scenario.\n")
                file.flush()
            except Exception as e:
                file.write(f"{scenario.scenario_id} fails. Jump this scenario.\n")
                file.flush()
        
        # success
        file.write("Success.\n")

if __name__ == "__main__":
    NUM_PROCESS = 10
    for i in range(80, 150, NUM_PROCESS):

        batch_processes: list[multiprocessing.Process] = []
        for j in range(NUM_PROCESS):
            p = multiprocessing.Process(target=generate_a_file, args=(i+j,))
            batch_processes.append(p)
            p.start()

        for p in batch_processes:
            p.join()
