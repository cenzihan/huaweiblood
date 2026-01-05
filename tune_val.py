import pandas as pd
from ultralytics import YOLO
import itertools
import torch

# 1. 指定你的最佳模型路径
# 这里填刚才的冠军权重 (Augmentation Tuning Winner)
MODEL_PATH = '/data/user/cenzihan/huaweiblood/huaweiblood/runs/segment/aug_yolov8m-seg_lr0.0_ud0.0_mix0.2/weights/best.pt'

# 2. 定义参数网格
# conf: 置信度阈值。对于 mAP 评分，通常低置信度能带来更高的 Recall，从而提高 mAP
# 但比赛可能有 FP (假阳性) 惩罚，所以要寻找平衡点
CONF_LIST = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

# iou: NMS 阈值。血小板可能密集重叠，IOU 阈值不能太低，否则会把紧挨着的细胞当成同一个抑制掉
IOU_LIST  = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

GRID = list(itertools.product(CONF_LIST, IOU_LIST))

def run_tuning():
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    results_list = []
    print(f"Total combinations: {len(GRID)}")

    for i, (conf, iou) in enumerate(GRID):
        print(f"[{i+1}/{len(GRID)}] Validating conf={conf}, iou={iou} ...")
        
        try:
            # 运行验证模式
            metrics = model.val(
                data='blood_cell.yaml',
                conf=conf,
                iou=iou,
                imgsz=640,       # 保持和训练一致
                plots=False,     # 只要分数，不画图，速度快
                verbose=False,
                device=0
            )
            
            map50 = metrics.seg.map50
            map50_95 = metrics.seg.map
            
            print(f"   >>> Result: mAP50-95={map50_95:.4f}")
            
            results_list.append({
                'conf': conf,
                'iou': iou,
                'mAP50': map50,
                'mAP50-95': map50_95
            })
            
        except Exception as e:
            print(f"   Error: {e}")

    # 汇总
    if results_list:
        df = pd.DataFrame(results_list)
        df = df.sort_values(by='mAP50-95', ascending=False)
        
        print("\n" + "="*60)
        print("🏆 Inference Tuning Leaderboard")
        print("="*60)
        print(df.head(10).to_string(index=False)) # 只看前10
        
        csv_path = 'tuning_val_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved full results to {csv_path}")
        
        best = df.iloc[0]
        print(f"\n✅ Optimal Params: conf={best['conf']}, iou={best['iou']}")
        print(f"   Expected Score: {best['mAP50-95']:.4f}")

if __name__ == '__main__':
    run_tuning()


