# YOLO 模型训练中心

一个完全运行在 Docker 中的 YOLO 批量训练平台：通过网页上传带标注的数据集，使用 GPU 执行训练，在网页查看任务进度，并下载训练得到的 `best.pt` 模型文件。

## 功能

- 中文网页控制台，显示 GPU Worker、排队任务、训练进度和已完成模型
- 上传 YOLO 格式标注数据集 ZIP
- 自动校验 `data.yaml`、图片、TXT 标签、类别编号和归一化坐标
- Redis 队列保证网页 API 与 GPU 训练进程解耦
- SQLite 保存任务状态，不需要单独部署 PostgreSQL
- 训练完成后提供 `best.pt`、`last.pt` 下载
- 支持取消任务、删除任务和自动清理临时文件
- 所有服务通过 Docker Compose 运行，不污染宿主机 Python/Node 环境

## 架构

```text
浏览器
  |
  v
yolo-web (Nginx + Vue, :8080)
  |
  v
yolo-api (FastAPI) ---> SQLite (/data/app.db)
  |
  v
Redis ---> yolo-worker (Ultralytics + NVIDIA GPU)

yolo-cleaner ---> 清理过期结果和临时文件
```

共 5 个容器：`yolo-web`、`yolo-api`、`yolo-worker`、`yolo-cleaner`、`redis`。SQLite 是 API 容器挂载的数据文件，不是独立容器。

## 环境要求

- Linux 主机
- Docker Engine 和 Docker Compose
- NVIDIA 驱动、NVIDIA Container Toolkit
- NVIDIA GPU（训练任务需要）
- 至少 20 GB 可用磁盘空间，具体取决于数据集和模型数量

当前部署使用已经验证过的 CUDA/Ultralytics 基础镜像 `yolo-batch-service:1.0`。因此当前 `Dockerfile` 适合已有该基础镜像的部署环境；在全新机器上构建前，需要准备同名基础镜像，或将 Dockerfile 的基础镜像替换为公开的 CUDA + Python 镜像并重新安装依赖。

## 快速启动

```sh
cp .env.example .env
openssl rand -hex 32
# 将生成的随机值填入 .env 的 API_KEY

docker compose --env-file .env up -d --build
docker compose --env-file .env ps
```

浏览器访问：`http://服务器地址:8080/`

网页右上角的 API Key 是本平台自己的接口访问凭据，不是 YOLO、Ultralytics 或 NVIDIA 的密钥。它必须与 `.env` 中的 `API_KEY` 完全一致。不要把 `.env` 提交到 GitHub。

## 数据集 ZIP 格式

训练 ZIP 至少应包含以下结构：

```text
dataset.zip
├── data.yaml
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   └── 0002.jpg
│   └── val/
│       └── 1001.jpg
└── labels/
    ├── train/
    │   ├── 0001.txt
    │   └── 0002.txt
    └── val/
        └── 1001.txt
```

图片和标签必须同名，例如 `images/train/0001.jpg` 对应 `labels/train/0001.txt`。标签每行格式为：

```text
class_id center_x center_y width height
```

坐标必须是 0 到 1 之间的归一化值。`data.yaml` 至少需要定义 `train`、`val` 和 `names`：

```yaml
path: .
train: images/train
val: images/val
names:
  0: object
```

网页选择 ZIP 后填写任务名称、基础模型、图片尺寸、Epoch 和 Batch Size，点击“开始创建训练任务”。训练成功后，在任务列表中点击下载即可取得 `best.pt`。

## 基础模型说明

默认的 `yolo11n.pt` 是预训练权重，用于迁移学习。它不是本次训练的最终结果，而是训练起点；训练完成后生成的 `best.pt` 才是针对上传数据集效果最好的模型。

如果完全从零训练，需要使用模型结构 YAML 并准备更长训练周期，通常不如使用预训练权重稳定，因此界面默认采用 `yolo11n.pt`。

## 数据生命周期和磁盘管理

数据目录位于 Compose 项目下的 `data/`：

- `uploads/`：原始 ZIP。任务成功处理后自动删除
- `work/`：解压后的临时训练数据。任务成功或失败后按流程清理
- `results/`：训练日志和结果文件。成功结果默认保留 30 天，失败任务默认保留 7 天
- `models/`：训练产出的模型文件。清理器不会自动删除，网页可手动删除任务及其模型
- `app.db`：SQLite 任务数据库

清理器默认每 5 分钟运行一次，并跳过排队中和运行中的任务。磁盘接近高水位时应优先在网页删除不再需要的任务和模型。

## API

API 位于 `/api/v1`，请求需要携带：

```http
X-API-Key: your-api-key
```

常用接口：

- `GET /api/v1/health`：健康检查
- `GET /api/v1/jobs`：任务列表
- `POST /api/v1/jobs`：上传训练 ZIP 并创建任务
- `POST /api/v1/jobs/{id}/cancel`：取消任务
- `GET /api/v1/jobs/{id}/download`：下载训练模型
- `DELETE /api/v1/jobs/{id}`：删除任务及相关文件

## 安全注意事项

- `.env` 只保存在服务器，不要提交到公开仓库
- 不要把真实服务器地址、SSH 密钥、API Key 或训练数据放入仓库
- 生产环境建议通过反向代理、内网访问控制或 VPN 暴露网页端口
- 公开部署前应更换默认 API Key，并限制上传文件大小和可访问网段

## 项目目录

```text
service/            FastAPI、SQLite、GPU Worker、清理器
web/                Vue 前端和 Nginx 静态站点
nginx/              Nginx 反向代理配置
docker-compose.yml  五容器编排
Dockerfile          API/Worker 应用镜像
requirements.txt    Python 依赖
scripts/             环境初始化脚本
```
