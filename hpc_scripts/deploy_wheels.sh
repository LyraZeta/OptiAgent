#!/bin/bash
# 请在本机设置环境变量，或直接在此处替换为真实值：
# export HPC_USER=your_username
# export HPC_HOST=your_hpc_address
HPC_TARGET="${HPC_USER}@${HPC_HOST}"

echo "🚀 正在把下载好的 PyMuPDF 和 pypdf 离线安装包上传到服务器..."

# 创建远程目录以防止路径不存在
ssh ${HPC_TARGET} "mkdir -p ~/OptiAgent/pymupdf_wheels/"

# 将离线包同步上传
rsync -avz --progress ./pymupdf_wheels/ ${HPC_TARGET}:~/OptiAgent/pymupdf_wheels/

echo "✅ 离线包已成功自动上传到超算！"
echo ""
echo "👉 请登录到超算（ssh ${HPC_TARGET}）"
echo "👉 运行以下命令进行离线安装："
echo "source ~/OptiAgent/my_optiagent_env/bin/activate && pip install --no-index -f ~/OptiAgent/pymupdf_wheels/ PyMuPDF pypdf"