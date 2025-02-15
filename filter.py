import os
import re

# 文件路径
folder_path = 'outputs/mapconvert_logs_validation/'  # 需要替换成你的文件夹路径

# 正则表达式模式
timeout_pattern = re.compile(r'(\w+) fails due to timeout\. Jump this scenario\.')
fails_pattern = re.compile(r'(\w+) fails\. Jump this scenario\.')
file_pattern = re.compile(r'0927-(\d{5})\.txt')

# 结果存储列表
timeout_results = []
fails_results = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 匹配文件名格式 0916-zzzzz.txt
    match_file = file_pattern.match(file_name)
    if match_file:
        zzzzz = match_file.group(1)
        file_path = os.path.join(folder_path, file_name)

        # 打开文件读取内容
        with open(file_path, 'r') as file:
            for line in file:
                # 匹配 xxxxx fails due to timeout. Jump this scenario.
                match_timeout = timeout_pattern.search(line)
                if match_timeout:
                    xxxxx = match_timeout.group(1)
                    timeout_results.append((int(zzzzz), xxxxx))

                # 匹配 yyyyy fails. Jump this scenario.
                match_fails = fails_pattern.search(line)
                if match_fails:
                    yyyyy = match_fails.group(1)
                    fails_results.append((int(zzzzz), yyyyy))

# 分别打印结果
print("Timeout Results (zzzzz, xxxxx):")
for result in timeout_results:
    print(result)

print("\nFails Results (zzzzz, yyyyy):")
for result in fails_results:
    print(result)
