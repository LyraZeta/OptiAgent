#!/bin/bash
# 请在本机设置环境变量，或直接在此处替换为真实值：
# export HPC_USER=your_username
# export HPC_HOST=your_hpc_address
HPC_TARGET="${HPC_USER}@${HPC_HOST}"

echo "🚀 正在把修改好的最新代码上传到服务器进行覆盖..."
rsync -avz --progress ./data_prep/ ${HPC_TARGET}:~/OptiAgent/data_prep/
# rsync -avz --progress ./tools/ ${HPC_TARGET}:~/OptiAgent/tools/
echo "✅ 代码已成功自动覆盖超算对应文件！"
