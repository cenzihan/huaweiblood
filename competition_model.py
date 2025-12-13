import os
import glob
import json
import torch
from ultralytics import YOLO

class Competition:
    def __init__(self):
        # 1. 自动选择设备: NPU > GPU > CPU
        self.device = self._get_device()
        print(f"Inference device: {self.device}")

        # 2. 加载模型
        # 注意: 提交时 best.pt 必须和此文件在同一级目录
        model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        if not os.path.exists(model_path):
            # 本地调试时的 fallback 路径，提交前请确保模型在根目录
            pass 
        
        self.model = YOLO(model_path, task='segment')

    def _get_device(self):
        # 优先检查华为 Ascend NPU
        try:
            import torch_npu
            from torch_npu.contrib import transfer_to_npu
            return 'npu'
        except ImportError:
            pass
        
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def get_results(self, test_path, output_path):
        """
        平台调用的主入口
        :param test_path: 测试集图片文件夹路径
        :param output_path: 结果输出文件夹路径
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 获取所有图片 (png 格式)
        image_files = glob.glob(os.path.join(test_path, '*.png'))
        
        # 批量推理参数
        predict_args = {
            'device': self.device,
            'imgsz': 1280,      # 关键: 提高分辨率检测小目标
            'conf': 0.15,       # 置信度阈值 (稍微调低以防漏检)
            'iou': 0.6,         # NMS 阈值
            'max_det': 300,     # 每张图最大目标数
            'verbose': False,   # 减少日志
            'augment': False    # 设为 True 可启用 TTA (Test Time Augmentation)，会变慢但可能更准
        }

        for img_path in image_files:
            filename = os.path.basename(img_path)
            json_name = os.path.splitext(filename)[0] + '.json'
            save_path = os.path.join(output_path, json_name)
            
            output_data = {
                "segmentation_classes": [],
                "segmentation_polygons": [],
                "segmentation_scores": []
            }

            try:
                # 推理单张
                results = self.model.predict(img_path, **predict_args)
                result = results[0] # 获取结果

                if result.masks is not None:
                    # 1. 类别 (全部转为字符串)
                    # 赛题只识别一类，ID为0
                    classes = result.boxes.cls.cpu().numpy().astype(int).astype(str)
                    output_data["segmentation_classes"] = classes.tolist()

                    # 2. 置信度
                    scores = result.boxes.conf.cpu().numpy()
                    output_data["segmentation_scores"] = scores.tolist()

                    # 3. 多边形坐标
                    # ultralytics 自动返回归一化后的坐标，无需 cv2 处理
                    # result.masks.xy 是一个 list, 每一项是 (N, 2) array
                    for poly in result.masks.xy:
                        output_data["segmentation_polygons"].append(poly.tolist())

            except Exception as e:
                print(f"Error processing {filename}: {e}")

            # 保存 JSON
            with open(save_path, 'w') as f:
                json.dump(output_data, f)