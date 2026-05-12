import os
import argparse
import re

def extract_number(s):
    base = os.path.splitext(s)[0]
    numbers = re.findall(r'\d+', base)
    return int(numbers[-1]) if numbers else float('inf')

def main():
    parser = argparse.ArgumentParser(description="批量自然数排序重命名文件")
    parser.add_argument('--Name', type=str, required=True, help='新文件基础名，如img')
    parser.add_argument('--Path', type=str, required=True, help='目标文件夹路径')
    args = parser.parse_args()

    dir_path = args.Path
    base_name = args.Name

    # 获取所有文件（不含目录），按文件名最后一个数字排序
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files_sorted = sorted(files, key=extract_number)

    num_files = len(files_sorted)
    num_digits = len(str(num_files))

    # 重命名
    for idx, old_name in enumerate(files_sorted, 0): ##这里的“0”指首个项目自然数为“0”
        ext = os.path.splitext(old_name)[1]
        new_name = f"{base_name}{str(idx).zfill(num_digits)}{ext}"
        old_path = os.path.join(dir_path, old_name)
        new_path = os.path.join(dir_path, new_name)
        # 如果新旧名一样跳过
        if old_name == new_name:
            continue
        os.rename(old_path, new_path)
        print(f"{old_name} --> {new_name}")

if __name__ == '__main__':
    main()
