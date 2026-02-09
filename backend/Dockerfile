FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（curl 用于健康检查，libgomp1 用于 PyTorch OpenMP 支持）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 训练模型（构建时执行，使用较少 epoch 加快构建）
RUN python train.py --epochs 3

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
