from ultralytics import YOLO

def train():
    # 1. 加载预训练的分割模型
    # YOLO11 模型系列:
    # yolo11n-seg.pt (nano)
    # yolo11s-seg.pt (small) - 推荐
    # yolo11m-seg.pt (medium)
    # yolo11l-seg.pt (large)
    model = YOLO('yolo11m-seg.pt') 

    # 2. 训练
    results = model.train(
        data='blood_cell.yaml',  # 数据集配置文件
        epochs=100,              # 训练轮数
        imgsz=640,               # 训练分辨率
        batch=32,                # 11s 比较轻，batch 可以开大点
        patience=20,             # 早停轮数
        
        # 关键优化：根据之前的扫参结果，关闭 mosaic 增强
        # 或者设为 0.0 直接关闭，或者用 close_mosaic=100
        mosaic=0.0,              
        
        device=5,                # GPU 设备 ID (保持你原设的 5)
        workers=8,
        name='yolo11s_blood_cell' # 更新实验名称
    )

if __name__ == '__main__':
    train()
