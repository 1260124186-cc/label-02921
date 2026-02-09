# 手写数字识别项目

基于 PyTorch CNN 的手写数字识别系统，使用 MNIST 数据集训练，提供 Web 画板交互界面。

## 快速启动

```bash
docker-compose up --build -d
```

启动后访问: http://localhost:8081

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
