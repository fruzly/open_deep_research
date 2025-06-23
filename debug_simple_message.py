#!/usr/bin/env python3
"""
简单的消息处理测试
"""

import sys
import os
sys.path.append('src')

from open_deep_research.message_manager import validate_and_fix_messages

def test_message_validation():
    """测试消息验证功能"""
    
    # 测试正常消息
    normal_msg = [{"role": "user", "content": "Write a short report about Python."}]
    print("🧪 测试正常消息:")
    print(f"  输入: {normal_msg}")
    
    fixed, fixes = validate_and_fix_messages(normal_msg, "google_genai")
    print(f"  输出: {fixed}")
    print(f"  修复: {fixes}")
    
    # 测试缺少role的消息
    broken_msg = [{"content": "Write a short report about Python."}]
    print("\n🧪 测试缺少role的消息:")
    print(f"  输入: {broken_msg}")
    
    fixed, fixes = validate_and_fix_messages(broken_msg, "google_genai")
    print(f"  输出: {fixed}")
    print(f"  修复: {fixes}")
    
    # 测试空消息列表
    empty_msg = []
    print("\n🧪 测试空消息列表:")
    print(f"  输入: {empty_msg}")
    
    fixed, fixes = validate_and_fix_messages(empty_msg, "google_genai")
    print(f"  输出: {fixed}")
    print(f"  修复: {fixes}")
    
    # 测试LangChain消息对象
    from langchain_core.messages import HumanMessage
    lc_msg = HumanMessage(content="Write a short report about Python.")
    print(f"\n🧪 测试LangChain消息对象:")
    print(f"  输入类型: {type(lc_msg)}")
    print(f"  输入内容: {lc_msg}")
    print(f"  输入属性: {lc_msg.__dict__ if hasattr(lc_msg, '__dict__') else 'N/A'}")
    
    # 尝试转换为字典
    try:
        lc_msg_dict = {"role": "user", "content": lc_msg.content}
        print(f"  转换为字典: {lc_msg_dict}")
        
        fixed, fixes = validate_and_fix_messages([lc_msg_dict], "google_genai")
        print(f"  修复结果: {fixed}")
        print(f"  修复操作: {fixes}")
    except Exception as e:
        print(f"  转换失败: {e}")

if __name__ == "__main__":
    test_message_validation() 