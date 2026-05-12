import os
import argparse
import shutil
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="对序列集图片进行按步长抽帧（精简）处理。")
    
    parser.add_argument(
        '-i', '--input_path', 
        type=str, 
        required=True, 
        help='输入数据集的根目录路径'
    )
    parser.add_argument(
        '-o', '--output_path', 
        type=str, 
        required=True, 
        help='输出数据集的根目录路径'
    )
    parser.add_argument(
        '-s', '--stride', 
        type=int, 
        default=2, 
        help='抽帧步长。例如：2 表示每2帧取1帧 (即取第0, 2, 4...帧)'
    )
    
    return parser.parse_args()

def process_frames(input_root, output_root, stride):
    # 计数器
    total_files_copied = 0
    total_dirs_processed = 0

    print(f"开始处理...")
    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print(f"步长 (Stride): {stride}")

    # 遍历输入目录
    for root, dirs, files in os.walk(input_root):
        # 筛选符合 frame_00XXX.png 格式的文件
        # 这里假设只要是以 png 结尾的文件均需处理
        image_files = [f for f in files if f.endswith('.JPG')]
        
        if not image_files:
            continue

        # 关键步骤：必须排序，确保按文件名顺序抽帧 (00001, 00002, ...)
        image_files.sort()

        # 计算相对路径，用于保持目录结构
        # 例如 root 是 /data/input/seq1/
        # rel_path 就是 seq1/
        rel_path = os.path.relpath(root, input_root)
        
        # 拼接输出目录
        target_dir = os.path.join(output_root, rel_path)
        
        # 创建目标文件夹（如果不存在）
        os.makedirs(target_dir, exist_ok=True)

        # 进行切片操作 (Downsampling)
        # list[::stride] 会从0开始，每隔 stride 取一个
        selected_frames = image_files[::stride]

        # 复制文件
        for frame in selected_frames:
            src_file = os.path.join(root, frame)
            dst_file = os.path.join(target_dir, frame)
            
            try:
                shutil.copy2(src_file, dst_file) # copy2 保留文件元数据（时间戳等）
                total_files_copied += 1
            except Exception as e:
                print(f"[错误] 复制文件失败: {src_file} -> {e}")

        total_dirs_processed += 1
        # 打印进度提示（可选，防止处理大量文件时以为卡死）
        print(f"已处理目录: {rel_path} | 原帧数: {len(image_files)} -> 抽取后: {len(selected_frames)}")

    print("-" * 30)
    print(f"处理完成！")
    print(f"共处理序列目录: {total_dirs_processed}")
    print(f"共输出图片文件: {total_files_copied}")
    print(f"输出路径: {output_root}")

if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"错误: 输入路径不存在 -> {args.input_path}")
        sys.exit(1)
        
    process_frames(args.input_path, args.output_path, args.stride)