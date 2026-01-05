import pandas as pd
from ultralytics import YOLO
import gc
import torch
import itertools

# 定义参数空间
# 只考虑 M 和 L 模型
MODELS = ['yolov8m-seg.pt', 'yolov8l-seg.pt']

# 1. 左右翻转 (0.0=关, 0.5=标准)
FLIPLR = [0.0, 0.5]

# 2. 上下翻转 (0.0=关, 0.5=标准)
# 对于细胞这种无方向性的物体，上下翻转通常很有用
FLIPUD = [0.0, 0.5]

# 3. 高斯噪声 (Gaussian Noise)
# YOLOv8 没有直接的 'noise' 参数，但可以通过 Albumentations 实现。
# 为了脚本简单，这里我们用 'mixup' (混合) 和 'hsv_h' (色调抖动) 来模拟干扰。
# 如果想严格加高斯噪声，需要改 yaml 配置，这里用 mixup 代替作为一种强干扰增强。
MIXUP  = [0.0, 0.2]

# 生成所有组合 (2*2*2*2 = 16组)
GRID = list(itertools.product(MODELS, FLIPLR, FLIPUD, MIXUP))

def run_tuning():
    results_list = []
    print(f"Total experiments: {len(GRID)}")

    for i, (model_name, fliplr, flipud, mixup) in enumerate(GRID):
        # 实验命名
        exp_name = f"aug_{model_name.split('.')[0]}_lr{fliplr}_ud{flipud}_mix{mixup}"
        print(f"\n[{i+1}/{len(GRID)}] Running: {model_name} | LR={fliplr} | UD={flipud} | Mix={mixup}")

        try:
            model = YOLO(model_name)

            # 训练参数
            # 固定 imgsz=640, 关闭 mosaic (基于之前的经验)
            model.train(
                data='blood_cell.yaml',
                epochs=100,               # 40轮快速验证
                imgsz=,
                batch=16 if 'm' in model_name else 8, # L模型显存大，减小batch
                
                # 核心扫参变量
                fliplr=fliplr,
                flipud=flipud,
                mixup=mixup,
                
                # 固定配置
                mosaic=0.0,              # 之前验证关闭 mosaic 效果好
                
                patience=10,
                device=0,
                workers=4,
                name=exp_name,
                exist_ok=True,
                verbose=False
            )

            # 获取结果
            metrics = model.metrics
            if metrics:
                map50 = metrics.seg.map50
                map50_95 = metrics.seg.map
                best_pt = str(model.trainer.best)
            else:
                map50 = 0.0
                map50_95 = 0.0
                best_pt = "Failed"

            print(f"   >>> Result: mAP50-95={map50_95:.4f}")

            results_list.append({
                'model': model_name,
                'fliplr': fliplr,
                'flipud': flipud,
                'mixup': mixup,
                'mAP50': map50,
                'mAP50-95': map50_95,
                'best_weights': best_pt
            })

            # 内存清理
            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"   !!! Failed: {e}")
            results_list.append({
                'model': model_name,
                'fliplr': fliplr,
                'flipud': flipud,
                'mixup': mixup,
                'mAP50': -1,
                'mAP50-95': -1,
                'best_weights': str(e)
            })

    # 汇总
    if results_list:
        df = pd.DataFrame(results_list)
        df = df.sort_values(by='mAP50-95', ascending=False)
        print("\n" + "="*80)
        print("🏆 Augmentation Tuning Leaderboard")
        print("="*80)
        print(df.to_string(index=False))
        df.to_csv('tuning_aug_results.csv', index=False)
        
        # 打印最佳
        best = df.iloc[0]
        print(f"\n✅ Best Config: {best['model']}, LR={best['fliplr']}, UD={best['flipud']}, Mix={best['mixup']}")

if __name__ == '__main__':
    run_tuning()

