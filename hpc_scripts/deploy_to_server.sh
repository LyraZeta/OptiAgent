#!/bin/bash
# 请在本机设置环境变量，或直接在此处替换为真实值：
# export HPC_USER=your_username
# export HPC_HOST=your_hpc_address
HPC_TARGET="${HPC_USER}@${HPC_HOST}"

echo "🗜️ 正在打包整个项目 (排除多余缓存/环境)..."
rsync -avz --exclude '.git' --exclude 'miniconda3' --exclude '__pycache__' --exclude 'data/chroma_db/' -e ssh . ${HPC_TARGET}:~/OptiAgent

echo "✅ 项目已全量同步到 ${HPC_TARGET}:~/OptiAgent"
