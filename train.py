from ultralytics import YOLO

def train():
    # 1. 加载预训练的分割模型
    # yolov8n-seg.pt (nano, 最快)
    # yolov8s-seg.pt (small)
    # yolov8m-seg.pt (medium, 推荐平衡点)
    # yolov8l-seg.pt (large, 最准但最慢)
    model = YOLO('yolov8m-seg.pt') 

    # 2. 训练
    results = model.train(
        data='blood_cell.yaml',  # 数据集配置文件
        epochs=100,              # 训练轮数
        imgsz=640,               # 训练分辨率 (如果显存够，建议设为 1024)
        batch=16,                # 批次大小
        patience=20,             # 早停轮数
        device=0,                # GPU 设备 ID
        workers=8,
        name='blood_cell_seg'    # 实验名称
    )

if __name__ == '__main__':
    train()