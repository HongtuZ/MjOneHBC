import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_qpos(path: str) -> tuple[list[str], np.ndarray]:
    """读取回放关节角度 npz，返回 (joint_names, qpos)。qpos 形状 (N, D)。"""
    data = np.load(path)
    return data["joint_names"].tolist(), data["qpos"]


def main():
    parser = argparse.ArgumentParser(description="对比 mj / real 回放的关节角度轨迹")
    parser.add_argument("--mj", type=str, required=True, help="mj_replay 保存的关节角度 npz")
    parser.add_argument("--real", type=str, required=True, help="real_replay 保存的关节角度 npz")
    parser.add_argument("--record", type=str, default=None, help="原始 action 记录 npz（计算期望 target 曲线）")
    parser.add_argument("--out", type=str, default="replay_diff.png", help="输出对比图路径")
    parser.add_argument("--start", type=int, default=0, help="绘图区间起点（步，含）")
    parser.add_argument("--end", type=int, default=None, help="绘图区间终点（步，不含；默认到最后）")
    args = parser.parse_args()

    mj_names, mj_qpos = load_qpos(args.mj)
    real_names, real_qpos = load_qpos(args.real)

    # 读取原始 action 记录，计算期望 target（target = action * scale + default_pos）
    target = None
    target_idx: dict[str, int] = {}
    if args.record:
        rec = np.load(args.record)
        actions = rec["actions"]
        action_scale = rec["action_scale"]
        default_joint_pos = rec["default_joint_pos"]
        target = actions * action_scale + default_joint_pos
        target_names = rec["joint_names"].tolist()
        target_idx = {name: i for i, name in enumerate(target_names)}

    # 以 mj 的关节名为基准，real 侧按名称对齐
    real_idx = {name: i for i, name in enumerate(real_names)}
    missing = [name for name in mj_names if name not in real_idx]
    if missing:
        print(f"[WARN] real 记录缺少关节: {missing}")

    # ---- 确定绘图区间，并裁剪数据 ----
    lens = [len(mj_qpos), len(real_qpos)]
    if target is not None:
        lens.append(len(target))
    n_max = min(lens)

    start = max(0, args.start)
    end = n_max if args.end is None else min(args.end, n_max)
    if end <= start:
        raise SystemExit(f"[ERROR] 区间无效: start={start} >= end={end}（数据总长 {n_max}）")

    s = slice(start, end)
    mj_qpos = mj_qpos[s]
    real_qpos = real_qpos[s]
    if target is not None:
        target = target[s]
    steps = np.arange(start, end)

    print(f"[DIFF] 绘图区间 step=[{start}, {end})，共 {end - start} 步")

    # ---- 画图 ----
    D = len(mj_names)
    ncols = 4
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4, nrows * 2.4), squeeze=False
    )

    for i, name in enumerate(mj_names):
        ax = axes[i // ncols][i % ncols]
        ax.plot(
            steps,
            mj_qpos[:, i],
            label="mj",
            color="tab:blue",
            linewidth=1.0,
        )
        if name in real_idx:
            j = real_idx[name]
            ax.plot(
                steps,
                real_qpos[:, j],
                label="real",
                color="tab:red",
                linewidth=1.0,
            )
        if target is not None and name in target_idx:
            j = target_idx[name]
            ax.plot(
                steps,
                target[:, j],
                label="target",
                color="tab:green",
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
            )
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("step", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)

    # 隐藏多余子图
    for j in range(D, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle(
        f"Replay joint angle comparison (step [{start}, {end}))\n"
        f"mj={Path(args.mj).name}  real={Path(args.real).name}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150)
    print(f"[DIFF] 已保存对比图到 {args.out}")


if __name__ == "__main__":
    main()
