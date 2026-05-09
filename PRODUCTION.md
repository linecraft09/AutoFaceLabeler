# AutoFaceLabeler — 生产环境执行指南

> 本机环境：WSL2 (Linux x64) · RTX 4070 Ti · CUDA 12.4 · Mihomo 代理

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| OS | Linux (本机 WSL2) |
| GPU | NVIDIA GPU + CUDA ≥ 11.8 |
| Python | 3.10 (conda env `afl`) |
| 代理 | Mihomo/Clash @ `127.0.0.1:7890`（YouTube 必需） |
| yt-dlp | ≥ 2026.03（已安装） |
| Deno | JS runtime（yt-dlp JS challenge 必需） |
| Cookies | YouTube Netscape 格式 cookies 文件 |

---

## 2. 环境变量 (.env)

项目根目录下 `.env` 文件（不入 git）：

```bash
# 代理（YouTube 必需）
HTTPS_PROXY=http://127.0.0.1:7890
HTTP_PROXY=http://127.0.0.1:7890

# LLM API Key（DeepSeek）
OPENAI_API_KEY=sk-your-key-here

# YouTube cookies 路径
YOUTUBE_COOKIES_PATH=config/youtube_cookies.txt
```

`.env.example` 已提供模板，可参考。

---

## 3. 配置文件 (config/config.yaml)

关键配置项说明：

```yaml
search:
  platforms:
    youtube:
      cookies: 'config/youtube_cookies.txt'
    bilibili:
      cookies: 'config/bilibili_cookies.txt'
  proxy: 'http://127.0.0.1:7890'       # 代理
  per_platform_results: 5              # 每平台每次搜索数

download:
  format: 'best[height<=720]'          # 限制分辨率加速下载
  platforms:
    youtube:
      cookies: 'config/youtube_cookies.txt'
    bilibili:
      cookies: 'config/bilibili_cookies.txt'
  proxy: 'http://127.0.0.1:7890'
  outtmpl: '/tmp/afl_raw/%(id)s/%(title)s [%(id)s].%(ext)s'

pipeline:
  target_qualified: 10                 # 目标合格视频数
  batch_size: 5                        # 每批处理数

project:
  data_dir: /data/afl_data             # V2 处理数据
  video_db: /data/afl_videos.db        # SQLite 数据库
  logs_dir: /data/afl_logs             # 日志目录

explorer:
  llm:
    enabled: false                     # LLM 搜索词优化（可选）
    api_key_env: OPENAI_API_KEY
    base_url: https://api.deepseek.com
    model: deepseek-chat
```

**注意**：生产环境建议修改 `project.*` 路径为非 `/tmp` 持久化目录。

---

## 4. YouTube Cookies 准备

1. 从浏览器导出 YouTube cookies（Netscape 格式）
2. 保存为 `config/youtube_cookies.txt`
3. 设置为只读：`chmod 444 config/youtube_cookies.txt`
4. Bilibili 使用独立的可写 visitor cookie 文件：`config/bilibili_cookies.txt`
5. 两个 cookie 文件都已在 `.gitignore` 中排除

**获取方式**：浏览器安装 "Get cookies.txt LOCALLY" 扩展 → 访问 YouTube → 导出。

**格式要求**：Netscape tab-separated，每行一个 cookie：

```
.youtube.com	TRUE	/	TRUE	1234567890	VISITOR_INFO1_LIVE	xxxxx
.youtube.com	TRUE	/	TRUE	1234567890	LOGIN_INFO	xxxxx
...
```

> HTTP header 格式（分号分隔单行）**不兼容**，需转换为 Netscape 格式。

---

## 5. 运行管道

### 5.1 激活环境

```bash
source /root/miniconda3/bin/activate afl
cd /root/codex_workspace/AutoFaceLabeler
```

### 5.2 直接运行

```bash
python src/core/pipeline_orchestrator.py --config config/config.yaml
```

### 5.3 运行集成测试（验证全链路）

```bash
python tests/integration_test.py
```

测试配置在 `tests/config_test.yaml`，使用 `/tmp/` 路径，适合快速验证。

### 5.4 通过 Codex 运行（带 GPU）

```bash
HTTPS_PROXY=http://127.0.0.1:7890 \
codex exec --sandbox danger-full-access \
  "/root/miniconda3/bin/conda run -n afl python tests/integration_test.py"
```

> ⚠️ `codex exec` 默认 sandbox 为 `workspace-write`（无法访问 conda/GPU）。
> 必须显式 `--sandbox danger-full-access` + `HTTPS_PROXY` 环境变量。

---

## 6. 管道流程

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌─────────────┐
│ Search  │───→│ V1 Filter│───→│ Download │───→│ V2 Coarse │───→│ V2 Fine  │───→│ Qualified ✓ │
│(yt-dlp) │    │(分辨率/时长)│   │(yt-dlp)  │    │(YOLO检测) │    │(ArcFace) │    │(保存+索引)  │
└─────────┘    └──────────┘    └──────────┘    └───────────┘    └──────────┘    └─────────────┘
      ↑                                                              │
      └──────────── 循环直到 target_qualified ────────────────────────┘
```

各阶段说明：

| 阶段 | 功能 | GPU |
|------|------|-----|
| Search | yt-dlp 搜索 YouTube/Bilibili，获取视频元数据 | ❌ |
| V1 Filter | 分辨率 ≥ 360p、时长 10-1800s、过滤黑名单频道 | ❌ |
| Download | 下载视频文件 | ❌ |
| V2 Coarse | YOLO 人脸检测 → 裁剪单人片段 | ✅ |
| V2 Fine | ArcFace 嵌入 + FAISS 去重 → 合格判定 | ✅ |

---

## 7. 输出结构

```
/tmp/afl_raw/                    # 下载的原始视频
  └── {video_id}/
        ├── {title}.mp4          # 原始视频
        └── {title}_clipped_0.mp4 # V2 裁剪片段

/tmp/afl_qualified_test/         # 合格视频（集成测试用）
  └── {video_id}_qualified.mkv

/tmp/afl_videos_test.db          # SQLite 状态数据库
/tmp/afl_test_faces.faiss        # FAISS 人脸索引
/tmp/afl_integration_test.jsonl  # 管道日志（JSON Lines）
```

**生产环境**应配置到持久化路径（不在 `/tmp`）。

---

## 8. 监控

### 8.1 实时管道日志

```bash
tail -f /tmp/afl_integration_test.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if d['event'] == 'stage':
        print(f\"[{d['stage']}] {d}\")
"
```

###### 8.1 实时管道日志

```bash
tail -f /tmp/afl_integration_test.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if d['event'] == 'stage':
        print(f\"[{d['stage']}] {d}\")
"
```

### 8.2 GPU 监控

```bash
watch -n 2 nvidia-smi
```

### 8.3 数据库查询

```bash
conda run -n afl python -c "
import sqlite3
db = sqlite3.connect('/tmp/afl_videos_test.db')
cur = db.execute('SELECT video_id, platform, status FROM videos ORDER BY rowid')
for row in cur:
    print(f'{row[0]:20s} | {row[1]:10s} | {row[2]}')
cur2 = db.execute(\"SELECT COUNT(*) FROM videos WHERE status='v2_passed'\")
print(f\"\nQualified: {cur2.fetchone()[0]}\")
"
```

---

## 9. 故障排查

### 9.1 YouTube 搜索/下载失败

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| `TLS handshake EOF` | 代理未生效 | 检查 `HTTPS_PROXY` 环境变量，确认 Mihomo 运行中 |
| `No video formats found` | cookies 失效/格式错误 | 重新导出 Netscape 格式 cookies |
| `Sign in to confirm` | IP 被标记 | 更换代理节点或 cookies |
| `HTTP Error 412` | Bilibili 反爬 | B站偶尔限流，自动重试即可 |

### 9.2 V2 不使用 GPU

```bash
conda run -n afl python -c "import torch; print(torch.cuda.is_available())"
```

如果返回 `False`：
- 检查 CUDA 版本与 PyTorch 匹配
- 检查 `nvidia-smi` 是否正常

### 9.3 yt-dlp JS Challenge 失败

确保 Deno 已安装：
```bash
which deno || echo "需要安装 Deno"
```

yt-dlp 配置中需包含：
```yaml
remote_components: ['ejs:github']
```

### 9.4 下载速度过慢

- 调整 `download.format` 为 `best[height<=720]` 或 `best[height<=480]`
- 检查代理节点质量
- Bilibili 视频通常不需要代理，可直连

---

## 10. 清理

```bash
# 清理测试残留
rm -rf /tmp/afl_raw /tmp/afl_data_test /tmp/afl_videos_test.db \
       /tmp/afl_explorer_test.json /tmp/afl_logs_test \
       /tmp/afl_integration_test.jsonl /tmp/afl_qualified_test

# 清理 FAISS 索引
rm -f /tmp/afl_test_faces.faiss
```

---

## 11. 快速启动清单

- [ ] 确认 Mihomo 代理运行：`curl -x http://127.0.0.1:7890 -s -o /dev/null -w "%{http_code}" https://www.youtube.com` → 200
- [ ] 确认 conda 环境：`conda run -n afl python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] 确认 cookies 有效：`ls -l config/youtube_cookies.txt config/bilibili_cookies.txt`
- [ ] 确认 `.env` 已配置
- [ ] 确认 `config.yaml` 路径正确（生产环境不用 `/tmp`）
- [ ] 修改 `config.yaml` 中 `project.*` 路径为持久化目录
- [ ] 运行：`python src/core/pipeline_orchestrator.py --config config/config.yaml`
