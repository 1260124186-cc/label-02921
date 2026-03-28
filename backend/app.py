"""
手写数字/字母/中文识别 API 服务
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
import logging
from model import HandwritingCNN
import os
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="手写数字识别 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型
model = None
device = None
transform = None
model_loaded = False
dataset_config = None


def get_dataset_config(model_name):
    """根据模型文件名获取数据集配置"""
    if 'emnist_letters' in model_name:
        return {
            'num_classes': 26,
            'dataset_type': 'emnist_letters',
            'label_mapping': {i: chr(ord('A') + i) for i in range(26)},
            'normalize': ((0.1751,), (0.3231,)),
            'need_rotation': True
        }
    elif 'emnist_byclass' in model_name:
        return {
            'num_classes': 62,
            'dataset_type': 'emnist_byclass',
            'label_mapping': lambda x: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'[x],
            'normalize': ((0.1751,), (0.3231,)),
            'need_rotation': True
        }
    elif 'emnist_digits' in model_name:
        return {
            'num_classes': 10,
            'dataset_type': 'emnist_digits',
            'label_mapping': {i: str(i) for i in range(10)},
            'normalize': ((0.1751,), (0.3231,)),
            'need_rotation': True
        }
    elif 'mnist' in model_name or model_name == 'handwriting_model':
        return {
            'num_classes': 10,
            'dataset_type': 'mnist',
            'label_mapping': {i: str(i) for i in range(10)},
            'normalize': ((0.1307,), (0.3081,)),
            'need_rotation': False
        }
    else:
        # 默认使用数字识别
        return {
            'num_classes': 10,
            'dataset_type': 'mnist',
            'label_mapping': {i: str(i) for i in range(10)},
            'normalize': ((0.1307,), (0.3081,)),
            'need_rotation': False
        }


def load_model(model_path=None):
    global model, device, transform, model_loaded, dataset_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 查找可用的模型文件
    model_dir = 'saved_models'
    available_models = []
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith('.pth') and 'handwriting_model' in f:
                available_models.append(f)

    if model_path is None:
        if not available_models:
            model_loaded = False
            print("错误: 未找到任何模型文件，预测功能不可用")
            return
        # 优先选择字母模型，其次是综合模型，最后是数字模型
        for priority in ['emnist_letters', 'emnist_byclass', 'emnist_digits', 'mnist', '']:
            for model_file in available_models:
                if priority in model_file:
                    model_path = os.path.join(model_dir, model_file)
                    break
            if model_path:
                break

    if model_path is None:
        model_path = os.path.join(model_dir, available_models[0])

    # 根据模型文件名确定数据集配置
    model_name = os.path.basename(model_path).replace('.pth', '')
    dataset_config = get_dataset_config(model_name)

    print(f"加载模型: {model_path}")
    print(f"数据集类型: {dataset_config['dataset_type']}")
    print(f"类别数量: {dataset_config['num_classes']}")

    model = HandwritingCNN(num_classes=dataset_config['num_classes']).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model_loaded = True
        print("模型加载成功")
    else:
        model_loaded = False
        print("错误: 模型文件不存在，预测功能不可用")

    # 创建图像转换
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
    ]

    # EMNIST数据集需要旋转处理
    if dataset_config['need_rotation']:
        transform_list.extend([
            transforms.Lambda(lambda img: img.rotate(-90, expand=True)),
            transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(*dataset_config['normalize'])
    ])

    transform = transforms.Compose(transform_list)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """上传图片进行预测"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载，请确保模型文件存在")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"图像处理失败: {e}")
        raise HTTPException(status_code=400, detail="图像处理失败，请检查图片格式")

    try:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_idx].item()

            # 将索引映射为字符
            label_mapping = dataset_config['label_mapping']
            if callable(label_mapping):
                predicted_char = label_mapping(predicted_idx)
            else:
                predicted_char = label_mapping.get(predicted_idx, str(predicted_idx))
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        raise HTTPException(status_code=500, detail="模型预测失败")

    return {
        "character": predicted_char,
        "confidence": round(confidence * 100, 2),
        "dataset_type": dataset_config['dataset_type']
    }


@app.post("/predict/base64")
async def predict_base64(data: dict):
    """Base64 图片预测"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载，请确保模型文件存在")

    image_data = data.get("image", "")
    if not image_data:
        raise HTTPException(status_code=400, detail="缺少图像数据")

    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # 转灰度并反转（画板白底黑字 -> MNIST黑底白字）
        gray = image.convert('L')
        gray_array = 255 - np.array(gray)
        image = Image.fromarray(gray_array.astype('uint8'))
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"图像处理失败: {e}")
        raise HTTPException(status_code=400, detail="图像处理失败，请检查图片格式")

    try:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_idx].item()

            # 将索引映射为字符
            label_mapping = dataset_config['label_mapping']
            if callable(label_mapping):
                predicted_char = label_mapping(predicted_idx)
            else:
                predicted_char = label_mapping.get(predicted_idx, str(predicted_idx))
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        raise HTTPException(status_code=500, detail="模型预测失败")

    return {
        "character": predicted_char,
        "confidence": round(confidence * 100, 2),
        "dataset_type": dataset_config['dataset_type']
    }
