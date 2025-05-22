#!/usr/bin/env python
"""
调试脚本 - 用于检查环境和配置
运行方式: python debug.py
"""

import os
import sys
import json
import platform
import subprocess

def print_section(title):
    """打印带有分隔符的标题"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

def check_file_exists(filepath):
    """检查文件是否存在并打印内容摘要"""
    if os.path.exists(filepath):
        print(f"✅ 文件存在: {filepath}")
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                if len(content) > 500:
                    content = content[:250] + "\n...\n" + content[-250:]
                print(f"文件内容预览:\n{content}")
        except Exception as e:
            print(f"❌ 无法读取文件: {e}")
    else:
        print(f"❌ 文件不存在: {filepath}")

def main():
    """主函数 - 运行所有调试检查"""
    print_section("系统信息")
    print(f"Python 版本: {sys.version}")
    print(f"平台: {platform.platform()}")
    print(f"当前工作目录: {os.getcwd()}")
    
    print_section("环境变量")
    # 只打印关键的环境变量，避免泄露敏感信息
    important_vars = [
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "PHONE_NUMBER", 
        "PORT", "RAILWAY_STATIC_URL", "RAILWAY_PUBLIC_DOMAIN",
        "PYTHON_VERSION", "NIXPACKS_PYTHON_VERSION"
    ]
    
    for var in important_vars:
        value = os.getenv(var)
        if value:
            if var in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]:
                # 隐藏敏感信息，只显示是否存在
                print(f"{var}: [已设置]")
            else:
                print(f"{var}: {value}")
        else:
            print(f"{var}: [未设置]")
    
    print_section("文件检查")
    # 检查关键文件是否存在
    key_files = [
        "app.py", "requirements.txt", "Procfile", 
        ".railway.json", "runtime.txt", "Dockerfile"
    ]
    
    for file in key_files:
        check_file_exists(file)
    
    print_section("依赖检查")
    try:
        result = subprocess.run(
            ["pip", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"❌ 无法列出已安装的包: {e}")
    
    print_section("网络检查")
    try:
        # 简单的连接测试
        result = subprocess.run(
            ["curl", "-s", "https://api.twilio.com/"], 
            capture_output=True, 
            text=True
        )
        print(f"Twilio API连接测试: {'成功' if result.returncode == 0 else '失败'}")
    except Exception as e:
        print(f"❌ 网络检查失败: {e}")

if __name__ == "__main__":
    main() 