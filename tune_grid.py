import pandas as pd
from ultralytics import YOLO
import gc
import torch
import os

# 定义要扫描的参数网格
# 包含了 n/s/m/l 四个等级的模型
# 针对每个模型，测试不同的分辨率和数据增强策略
GRID = [
    # 1. Nano (速度最快，适合极小数据)
    {'model': 'yolov8n-seg.pt', 'imgsz': 640,  'batch': 32, 'mosaic': 1.0},
    {'model': 'yolov8n-seg.pt', 'imgsz': 1280, 'batch': 16, 'mosaic': 1.0},
    
    # 2. Small (平衡性好)
    {'model': 'yolov8s-seg.pt', 'imgsz': 640,  'batch': 32, 'mosaic': 1.0},
    {'model': 'yolov8s-seg.pt', 'imgsz': 1280, 'batch': 8,  'mosaic': 1.0},
    {'model': 'yolov8s-seg.pt', 'imgsz': 640,  'batch': 32, 'mosaic': 0.0}, # 关闭 mosaic 看看效果
    
    # 3. Medium (之前的冠军模型)
    {'model': 'yolov8m-seg.pt', 'imgsz': 640,  'batch': 16, 'mosaic': 1.0},
    {'model': 'yolov8m-seg.pt', 'imgsz': 1280, 'batch': 4,  'mosaic': 1.0},
    
    # 4. Large (大模型，容易过拟合但潜力大)
    {'model': 'yolov8l-seg.pt', 'imgsz': 640,  'batch': 8,  'mosaic': 1.0},
    {'model': 'yolov8l-seg.pt', 'imgsz': 1280, 'batch': 2,  'mosaic': 1.0},
]

def run_tuning():
    results_list = []
    total_exps = len(GRID)
    print(f"Total experiments to run: {total_exps}")

    for i, cfg in enumerate(GRID):
        model_name = cfg['model']
        imgsz = cfg['imgsz']
        batch = cfg['batch']
        mosaic = cfg['mosaic']
        
        # 实验名称包含关键参数
        exp_name = f"tune_{model_name.split('.')[0]}_{imgsz}_m{int(mosaic)}"
        
        print(f"\n[{i+1}/{total_exps}] Running: {model_name} @ {imgsz}px, Mosaic={mosaic}, Batch={batch} ...")

        try:
            # 1. 加载模型
            model = YOLO(model_name)

            # 2. 训练
            # 使用较短的 epochs (50) 快速验证
            # close_mosaic=10: 最后10轮关闭 mosaic 增强，有助于精调
            model.train(
                data='blood_cell.yaml',
                epochs=50,
                imgsz=imgsz,
                batch=batch,
                mosaic=mosaic,           # 数据增强参数
                close_mosaic=10,         
                patience=10,
                device=0,
                workers=4,
                name=exp_name,
                exist_ok=True,
                verbose=False
            )

            # 3. 获取最佳验证结果
            metrics = model.metrics
            if metrics:
                map50 = metrics.seg.map50
                map50_95 = metrics.seg.map
                
                # 获取最佳权重路径
                best_pt = str(model.trainer.best)
            else:
                map50 = 0.0
                map50_95 = 0.0
                best_pt = "Failed"

            print(f"   >>> Result: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}")

            results_list.append({
                'model': model_name,
                'imgsz': imgsz,
                'batch': batch,
                'mosaic': mosaic,
                'mAP50': map50,
                'mAP50-95': map50_95,
                'best_weights': best_pt
            })

            # 4. 激进清理显存
            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"   !!! Experiment failed: {e}")
            # 记录失败但继续下一个
            results_list.append({
                'model': model_name,
                'imgsz': imgsz,
                'batch': batch,
                'mosaic': mosaic,
                'mAP50': -1.0,
                'mAP50-95': -1.0,
                'best_weights': str(e)
            })

    # 5. 汇总与保存
    if not results_list:
        print("No results collected.")
        return

    df = pd.DataFrame(results_list)
    # 按 mAP50-95 降序排列
    df = df.sort_values(by='mAP50-95', ascending=False)
    
    print("\n" + "="*80)
    print("🏆 Tuning Leaderboard (Sorted by mAP50-95)")
    print("="*80)
    # 格式化打印
    print(df[['model', 'imgsz', 'mosaic', 'mAP50', 'mAP50-95']].to_string(index=False))
    
    csv_path = 'tuning_results_full.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to {csv_path}")
    
    # 打印冠军
    best = df.iloc[0]
    print(f"\n✅ Winner: {best['model']} @ {best['imgsz']}px (mAP50-95: {best['mAP50-95']:.4f})")
    print(f"   Weights: {best['best_weights']}")

if __name__ == '__main__':
    run_tuning()




