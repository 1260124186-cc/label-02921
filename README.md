# 手写数字识别项目

基于 PyTorch CNN 的手写数字识别系统，使用 MNIST 数据集训练，提供 Web 画板交互界面。

## How to Run

### 方式一：Docker 运行（推荐）

```bash
docker-compose up --build -d
```

启动后访问: http://localhost:8081

> 注：首次构建镜像时会自动下载 MNIST 数据集并训练模型（约 2-3 分钟），训练完成后模型会打包进镜像，后续启动无需重复训练。

### 方式二：本地直接运行

**1. 启动后端**

```bash
cd backend
pip install -r requirements.txt
python train.py          # 首次运行需要训练模型
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**2. 启动前端**

```bash
cd frontend-admin
python -m http.server 8081
```

访问: http://localhost:8081

> 注：本地运行前端时，需要修改 `index.html` 中的 API 地址，将 `/api/predict/base64` 改为 `http://localhost:8000/api/predict/base64`（因为没有 Nginx 反向代理）。

## Services

| 服务 | 端口 | 说明 |
|------|------|------|
| frontend-admin | 8081 | Web 前端界面 |
| backend | 8000 (内部) | FastAPI 后端 API |

## 测试账号

无需登录，直接使用。

## 题目内容

使用 Python 实现一个手写字体识别项目

---

## 项目介绍

基于 PyTorch CNN 的手写数字识别系统，使用 MNIST 数据集训练，提供 Web 画板交互界面。

### 技术栈

- 后端: Python + FastAPI + PyTorch
- 前端: HTML/CSS/JavaScript + Nginx
- 部署: Docker + Docker Compose

### 项目结构

```
├── backend/              # 后端服务
│   ├── app.py            # FastAPI 应用
│   ├── Dockerfile
│   ├── model.py          # CNN 模型定义
│   ├── requirements.txt
│   └── train.py          # 模型训练脚本
├── frontend-admin/       # 前端服务
│   ├── Dockerfile
│   ├── index.html        # 画板交互界面
│   └── nginx.conf        # Nginx 反向代理配置
├── .gitignore
├── docker-compose.yml
└── README.md
```

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/predict` | POST | 上传图片文件预测 |
| `/api/predict/base64` | POST | Base64 图片预测 |

### 使用方式

1. 在画板上绘制数字 0-9
2. 点击「识别」按钮
3. 查看识别结果和置信度
