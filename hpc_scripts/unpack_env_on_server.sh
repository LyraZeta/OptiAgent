#!/bin/bash
echo "📦 正在超算上解压缩移植过来的本地环境，大概需要几十秒..."
mkdir -p my_optiagent_env
tar -xzf optiagent_env.tar.gz -C my_optiagent_env

echo "🔧 自动修复环境路径，使他在超算上也能完美执行..."
# 用 conda-pack 自带的安全恢复机制强制覆盖所有系统级旧路径链接
source my_optiagent_env/bin/activate
conda-unpack

echo "🎉 终极解压完成！在提交计算任务时，你的核心命令只有一句话："
echo "./my_optiagent_env/bin/python data_prep/parse_pdf.py"
