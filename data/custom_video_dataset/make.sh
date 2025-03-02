#!/bin/bash

# 遍历主目录下的每个子目录
for subdir in videos/*/; do
    (
        # 进入子目录，若失败则退出
        cd "$subdir" || { echo "无法进入目录 $subdir"; exit 1; }
        echo "处理目录: $subdir"

        # 步骤1: 生成videos.txt（包含video目录下所有.avi文件的相对路径）
        # 格式示例: video/v_xxxxx.avi
        if [ -d "video" ]; then
            # 递归查找video目录下的所有.avi文件，并写入videos.txt
            find video -type f -name "*.avi" | sort > videos.txt
            echo "已生成: $subdir/videos.txt"
        else
            echo "警告: 目录 $subdir 中不存在 video 目录"
        fi

        # 步骤2: 创建空文件prompts.txt
        touch prompts.txt
        echo "已创建: $subdir/prompts.txt"
    )
done

echo "全部操作完成！"