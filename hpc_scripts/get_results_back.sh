#!/bin/bash
# 请在本机设置环境变量，或直接在此处替换为真实值：
# export HPC_USER=your_username
# export HPC_HOST=your_hpc_address
HPC_TARGET="${HPC_USER}@${HPC_HOST}"

echo "📦 正在从服务器拉取生成的解析库到本地..."
rsync -avz -e ssh ${HPC_TARGET}:~/OptiAgent/data/parsed_md ./data/
rsync -avz -e ssh ${HPC_TARGET}:~/OptiAgent/data/chroma_db ./data/

echo "✅ 高质量数据已全部拉取回本地，可以运行 streamlit run app.py 测试了！"
