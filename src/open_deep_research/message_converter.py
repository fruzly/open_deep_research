"""
消息转换器模块
专门处理LangChain消息对象与字典格式之间的转换
"""

from typing import List, Dict, Any, Union
from langchain_core.messages import BaseMessage


def convert_langchain_messages_to_dict(messages: List[Union[BaseMessage, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    将LangChain消息对象转换为字典格式
    
    Args:
        messages: 混合格式的消息列表（可能包含LangChain对象和字典）
        
    Returns:
        统一的字典格式消息列表
    """
    dict_messages = []
    
    for msg in messages:
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            # This is a LangChain message object
            if msg.type == 'human':
                dict_messages.append({"role": "user", "content": msg.content})
            elif msg.type == 'ai':
                msg_dict = {"role": "assistant", "content": msg.content}
                # Preserve tool calls if present
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                dict_messages.append(msg_dict)
            elif msg.type == 'system':
                dict_messages.append({"role": "system", "content": msg.content})
            elif msg.type == 'tool':
                msg_dict = {"role": "tool", "content": msg.content}
                # Preserve tool call metadata if present
                if hasattr(msg, 'tool_call_id'):
                    msg_dict["tool_call_id"] = msg.tool_call_id
                if hasattr(msg, 'name'):
                    msg_dict["name"] = msg.name
                dict_messages.append(msg_dict)
            else:
                # Fallback to user role
                dict_messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, dict):
            # Already in dict format
            dict_messages.append(msg)
        else:
            # Unknown format, try to convert
            dict_messages.append({"role": "user", "content": str(msg)})
    
    return dict_messages 