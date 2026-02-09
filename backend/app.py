"""
手写数字识别 API 服务
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


def load_model():
    global model, device, transform, model_loaded
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwritingCNN().to(device)

    model_path = 'saved_models/handwriting_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model_loaded = True
        print("模型加载成功")
    else:
        model_loaded = False
        print("错误: 模型文件不存在，预测功能不可用")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


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
            predicted = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted].item()
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        raise HTTPException(status_code=500, detail="模型预测失败")

    return {
        "digit": predicted,
        "confidence": round(confidence * 100, 2)
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
            predicted = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted].item()
    except Exception as e:
        logger.error(f"模型预测失败: {e}")
        raise HTTPException(status_code=500, detail="模型预测失败")

    return {
        "digit": predicted,
        "confidence": round(confidence * 100, 2)
    }
