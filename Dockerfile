FROM python:3.9-slim

WORKDIR /app

# 复制项目文件
COPY . .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV PORT=5000

# 暴露端口
EXPOSE $PORT

# 启动命令
CMD gunicorn app:app --bind 0.0.0.0:$PORT 