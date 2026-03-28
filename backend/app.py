"""
手写体识别 API 服务
支持数字、字母、中文等多种识别模式
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
from model import HandwritingCNN, CharacterSet
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="手写体识别 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型字典
models = {}
transform = None
device = None


def load_all_models():
    """加载所有模型"""
    global models, transform, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_types = ['digits', 'emnist_balanced', 'chinese']

    for dataset_type in dataset_types:
        try:
            num_classes = CharacterSet.get_num_classes(dataset_type)
            model = HandwritingCNN(num_classes=num_classes).to(device)

            model_path = f'saved_models/handwriting_model_{dataset_type}.pth'
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                models[dataset_type] = model
                print(f"模型加载成功: {dataset_type}")
            else:
                print(f"模型文件不存在: {dataset_type}")
        except Exception as e:
            print(f"加载模型 {dataset_type} 失败: {e}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print(f"已加载模型: {list(models.keys())}")


def get_model(dataset_type='emnist_balanced'):
    """获取指定模型"""
    if dataset_type not in models:
        raise HTTPException(status_code=503, detail=f"模型 {dataset_type} 未加载，请先训练模型")
    return models[dataset_type]


@app.on_event("startup")
async def startup():
    load_all_models()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_models": list(models.keys()),
        "available_models": ['digits', 'emnist_balanced', 'chinese']
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), dataset_type: str = 'emnist_balanced'):
    """上传图片进行预测"""
    model = get_model(dataset_type)

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

            mapping = CharacterSet.get_mapping(dataset_type)
            character = mapping.get(predicted_idx, str(predicted_idx))
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        raise HTTPException(status_code=500, detail="模型预测失败")

    return {
        "character": character,
        "index": predicted_idx,
        "confidence": round(confidence * 100, 2),
        "dataset_type": dataset_type
    }


@app.post("/predict/base64")
async def predict_base64(data: dict):
    """Base64 图片预测"""
    dataset_type = data.get("dataset_type", "emnist_balanced")
    model = get_model(dataset_type)

    image_data = data.get("image", "")
    if not image_data:
        raise HTTPException(status_code=400, detail="缺少图像数据")

    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # 转灰度并反转（画板白底黑字 -> 黑底白字）
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

            mapping = CharacterSet.get_mapping(dataset_type)
            character = mapping.get(predicted_idx, str(predicted_idx))
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        raise HTTPException(status_code=500, detail="模型预测失败")

    return {
        "character": character,
        "index": predicted_idx,
        "confidence": round(confidence * 100, 2),
        "dataset_type": dataset_type
    }
