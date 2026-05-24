# 部署与运行说明

## 环境要求

- Python 3.11，推荐 Anaconda 环境。
- Node.js 20+，当前验证环境为 Node 24。
- Python 包：`pandas`、`numpy`、`scikit-learn`、`torch`、`fastapi`、`uvicorn`、`pytest`。
- 前端包：在 `dashboard/frontend` 下执行 `npm install`。

## 一键复现流程

```powershell
cd D:\part3data
python scripts/build_stage1_datasets.py
python scripts/validate_stage1_outputs.py
python scripts/train_stage2_model.py
python scripts/build_external_fusion_dataset.py
python scripts/train_safe_marl.py --epochs 120
node scripts/generate_competition_report.cjs
python -m pytest
```

## 启动后端

```powershell
cd D:\part3data
python -m uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
```

## 启动前端

```powershell
cd D:\part3data\dashboard\frontend
npm run dev
```

浏览器访问：

```text
http://127.0.0.1:5173
```

## 生产构建

```powershell
cd D:\part3data\dashboard\frontend
npm run build
```

构建产物在 `dashboard/frontend/dist/`。如需部署到 Nginx，可将 `/api` 反向代理到 FastAPI 后端。
