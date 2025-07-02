"""
Message Converter Module
Specializes in converting between LangChain message objects and dictionary format.
"""

from typing import List, Dict, Any, Union
from langchain_core.messages import BaseMessage
import structlog

# Configure logging
logger = structlog.get_logger(__name__)

def convert_langchain_messages_to_dict(messages: List[Union[BaseMessage, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Converts a list of LangChain message objects to a list of dictionaries.
    
    Args:
        messages: A list of messages in mixed formats (may include LangChain objects and dictionaries).
        
    Returns:
        A list of messages in a unified dictionary format.
    """
    logger.info(f"Starting message conversion - input message count: {len(messages)}")
    
    dict_messages = []
    
    for i, msg in enumerate(messages):
        try:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # This is a LangChain message object
                logger.debug(f"Processing LangChain message object - index: {i}, type: {msg.type}")
                
                content = msg.content if msg.content is not None else ''
                # Ensure content is never empty or just whitespace for Gemini API compatibility
                if not content or (isinstance(content, str) and not content.strip()):
                    content = " " # Provide a single space to ensure it's not empty
                    logger.debug(f"Provided single space for empty content - index: {i}, type: {msg.type}")

                content = str(content)
                
                # Original logic for providing more descriptive default content if still too short
                if len(content.strip()) < 2: # Check stripped length to avoid issues with just a space
                    has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                    if msg.type == 'ai' and has_tool_calls:
                        content = "Calling tools."
                    elif msg.type == 'human':
                        content = "Please continue."
                    elif msg.type == 'system':
                        content = "System prompt."
                    elif msg.type == 'tool':
                        content = "Tool execution completed."
                    else:
                        content = "Continue."
                    logger.debug(f"Provided more descriptive default content for short message - index: {i}, type: {msg.type}")
                
                if msg.type == 'human':
                    msg_dict = {"role": "user", "content": content}
                    dict_messages.append(msg_dict)
                    logger.debug(f"Converted human message - index: {i}, content length: {len(content)}")
                elif msg.type == 'ai':
                    msg_dict = {"role": "assistant", "content": content}
                    # Preserve tool calls if present
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        msg_dict["tool_calls"] = msg.tool_calls
                        logger.debug(f"Converted AI message - index: {i}, contains tool calls: {len(msg.tool_calls)}")
                    else:
                        logger.debug(f"Converted AI message - index: {i}, content length: {len(content)}")
                    dict_messages.append(msg_dict)
                elif msg.type == 'system':
                    msg_dict = {"role": "system", "content": content}
                    dict_messages.append(msg_dict)
                    logger.debug(f"Converted system message - index: {i}, content length: {len(content)}")
                elif msg.type == 'tool':
                    msg_dict = {"role": "tool", "content": content}
                    # Preserve tool call metadata if present
                    if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                        msg_dict["tool_call_id"] = msg.tool_call_id
                    if hasattr(msg, 'name') and msg.name:
                        msg_dict["name"] = msg.name
                    dict_messages.append(msg_dict)
                    logger.debug(f"Converted tool message - index: {i}, content length: {len(content)}")
                else:
                    # Fallback to user role
                    msg_dict = {"role": "user", "content": content}
                    dict_messages.append(msg_dict)
                    logger.warning(f"Unknown LangChain message type - index: {i}, type: {msg.type}, falling back to user role")
            elif isinstance(msg, dict):
                # Already in dict format, but ensure content is not None or empty
                logger.debug(f"Processing dictionary format message - index: {i}")
                msg_copy = msg.copy()
                
                # Ensure content field exists and is valid
                if 'content' not in msg_copy or msg_copy['content'] is None:
                    msg_copy['content'] = ''
                else:
                    msg_copy['content'] = str(msg_copy['content']).strip()
                
                # Ensure content is never empty for Gemini API compatibility
                if not msg_copy['content']:
                    role = msg_copy.get('role', 'user')
                    has_tool_calls = bool(msg_copy.get('tool_calls'))
                    
                    if role == 'assistant' and has_tool_calls:
                        msg_copy['content'] = "Calling tools."
                    elif role == 'user':
                        msg_copy['content'] = "Please continue."
                    elif role == 'system':
                        msg_copy['content'] = "System prompt."
                    elif role == 'tool':
                        msg_copy['content'] = "Tool execution completed."
                    else:
                        msg_copy['content'] = "Continue."
                    logger.debug(f"Provided default content for empty dictionary message - index: {i}, role: {role}")
                
                dict_messages.append(msg_copy)
            else:
                # Unknown format, convert to user message
                content = str(msg) if msg else "Please continue."
                dict_messages.append({"role": "user", "content": content})
                logger.warning(f"Unknown message format - index: {i}, type: {type(msg)}, converting to user message")
        except Exception as e:
            logger.error(f"Error during message conversion - index: {i}, error: {str(e)}")
            # Add a safe fallback message
            dict_messages.append({"role": "user", "content": "Please continue."})
    
    logger.info(f"Message conversion complete - output message count: {len(dict_messages)}")
    return dict_messages 