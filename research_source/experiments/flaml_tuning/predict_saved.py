"""Replay frozen final models without repeating the parameter search."""
# 这是日常复现预测的简短入口：读取参数 → 载入数据 → 重建特征/加载模型 → 保存 CSV。
# 不重新搜索参数，也不重新训练；真正的特征重建与模型调用在 verify_results.predict_saved 中。

# 运行参数和路径处理。
import argparse
from pathlib import Path

# 复用数据加载器、项目根目录和模型重放函数；仍需保留前序实验存档。
from experiments.data_processing_round1.run_experiment import load_data
from experiments.dart_dropout.run_experiment import REPO
from .verify_results import predict_saved


def main():
    # --run 指向已完成的实验目录（含 final_models），--data-dir 指向原始数据目录，
    # --output 指向要新建的预测 CSV。项目默认数据目录在仓库的上一级 data/。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=REPO.parent/'data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # 已存在的输出文件直接报错，避免覆盖之前的结果。
    if args.output.exists():
        raise FileExistsError(args.output)
    # 读取原始成交与隐藏测试数据，再用保存的最终模型重放完整房型方案。
    # 输出只有 ml_num 与 predicted_price 两列；这里不计算隐藏测试误差。
    raw, test = load_data(args.data_dir)
    result = predict_saved(args.run.resolve(), raw, test)
    # 自动建立输出文件的父目录，保存 CSV 且不附加 DataFrame 行号。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


# 作为模块运行时启动命令行入口；仅导入此文件时不生成预测。
if __name__ == '__main__':
    main()
