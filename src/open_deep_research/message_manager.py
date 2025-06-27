"""
Universal Message Manager - Supports all major LLM providers
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

# Configure logging
logger = structlog.get_logger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers - fully compatible with LangChain init_chat_model"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE_GENAI = "google_genai"  # Google GenAI
    GOOGLE_VERTEXAI = "google_vertexai"  # Google VertexAI
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    # New providers
    COHERE = "cohere"
    FIREWORKS = "fireworks"
    MISTRALAI = "mistralai"
    XAI = "xai"
    PERPLEXITY = "perplexity"
    BEDROCK = "bedrock"
    BEDROCK_CONVERSE = "bedrock_converse"
    TOGETHER = "together"
    IBM = "ibm"
    NVIDIA = "nvidia"
    AZURE_AI = "azure_ai"
    UNKNOWN = "unknown"

@dataclass
class ProviderRequirements:
    """Message sequence requirements for LLM providers"""
    must_start_with_user: bool = False
    allow_consecutive_same_role: bool = True
    requires_alternating: bool = False
    supports_system_anywhere: bool = True
    supports_tool_calls: bool = True
    max_consecutive_assistant: int = 999
    requires_user_after_tool: bool = False

class UniversalMessageManager:
    """Universal Message Manager, supports all major LLM providers"""
    
    # Message sequence requirements for each provider
    PROVIDER_REQUIREMENTS = {
        LLMProvider.OPENAI: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.AZURE_OPENAI: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.ANTHROPIC: ProviderRequirements(
            must_start_with_user=True,
            allow_consecutive_same_role=False,
            requires_alternating=True,
            supports_system_anywhere=False,
            supports_tool_calls=True
        ),
        LLMProvider.GOOGLE_GENAI: ProviderRequirements(
            must_start_with_user=True,
            allow_consecutive_same_role=False,
            requires_alternating=True,
            supports_system_anywhere=True,
            supports_tool_calls=True,
            max_consecutive_assistant=1
        ),
        LLMProvider.GOOGLE_VERTEXAI: ProviderRequirements(
            must_start_with_user=True,
            allow_consecutive_same_role=False,
            requires_alternating=True,
            supports_system_anywhere=True,
            supports_tool_calls=True,
            max_consecutive_assistant=1
        ),
        LLMProvider.HUGGINGFACE: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.OLLAMA: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.DEEPSEEK: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.GROQ: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.COHERE: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.FIREWORKS: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.MISTRALAI: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.XAI: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.PERPLEXITY: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.BEDROCK: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.BEDROCK_CONVERSE: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.TOGETHER: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.IBM: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.NVIDIA: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.AZURE_AI: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        ),
        LLMProvider.UNKNOWN: ProviderRequirements(
            must_start_with_user=False,
            allow_consecutive_same_role=True,
            supports_system_anywhere=True,
            supports_tool_calls=True
        )
    }
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize the message manager.
        
        Args:
            provider: The name of the LLM provider. If None, it will be auto-detected.
        """
        self.provider = self._detect_provider(provider) if provider else LLMProvider.UNKNOWN
        self.requirements = self.PROVIDER_REQUIREMENTS[self.provider]
        
        logger.info(f"Initializing message manager - Provider: {self.provider.value}, Requirements: {self.requirements}")
        
    def _detect_provider(self, provider_hint: str) -> LLMProvider:
        """
        Detects the LLM provider based on a hint.
        
        Follows the inference rules of LangChain's init_chat_model:
        https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
        """
        logger.debug(f"Starting LLM provider detection - hint: {provider_hint}")
        
        if not provider_hint:
            logger.warning("Provider hint is empty, returning UNKNOWN")
            return LLMProvider.UNKNOWN
            
        provider_hint = provider_hint.lower()
        
        # 1. First, check for explicit provider prefixes (e.g., "openai:gpt-4")
        if ':' in provider_hint:
            provider_prefix = provider_hint.split(':')[0]
            logger.debug(f"Detected provider prefix - prefix: {provider_prefix}")
            
            # Check for LangChain-supported provider prefixes
            provider_mapping = {
                'openai': LLMProvider.OPENAI,
                'anthropic': LLMProvider.ANTHROPIC,
                'azure_openai': LLMProvider.AZURE_OPENAI,
                'azure_ai': LLMProvider.AZURE_AI,
                'google_vertexai': LLMProvider.GOOGLE_VERTEXAI,
                'google_genai': LLMProvider.GOOGLE_GENAI,
                'bedrock': LLMProvider.BEDROCK,
                'bedrock_converse': LLMProvider.BEDROCK_CONVERSE,
                'cohere': LLMProvider.COHERE,
                'fireworks': LLMProvider.FIREWORKS,
                'together': LLMProvider.TOGETHER,
                'mistralai': LLMProvider.MISTRALAI,
                'huggingface': LLMProvider.HUGGINGFACE,
                'groq': LLMProvider.GROQ,
                'ollama': LLMProvider.OLLAMA,
                'deepseek': LLMProvider.DEEPSEEK,
                'ibm': LLMProvider.IBM,
                'nvidia': LLMProvider.NVIDIA,
                'xai': LLMProvider.XAI,
                'perplexity': LLMProvider.PERPLEXITY,
            }
            if provider_prefix in provider_mapping:
                detected_provider = provider_mapping[provider_prefix]
                logger.info(f"Provider detected by prefix - {detected_provider.value}")
                return detected_provider
        
        # 2. Follow LangChain's model name inference rules
        # GPT series -> OpenAI
        if any(provider_hint.startswith(prefix) for prefix in ['gpt-3', 'gpt-4', 'o1', 'o3']):
            logger.info(f"OpenAI detected by model name - model: {provider_hint}")
            return LLMProvider.OPENAI
        
        # Claude series -> Anthropic
        if provider_hint.startswith('claude'):
            logger.info(f"Anthropic detected by model name - model: {provider_hint}")
            return LLMProvider.ANTHROPIC
        
        # Amazon series -> Bedrock
        if provider_hint.startswith('amazon'):
            logger.info(f"Bedrock detected by model name - model: {provider_hint}")
            return LLMProvider.BEDROCK
        
        # Gemini series -> Google GenAI
        if provider_hint.startswith('gemini'):
            logger.info(f"Google GenAI detected by model name - model: {provider_hint}")
            return LLMProvider.GOOGLE_GENAI
        
        # Command series -> Cohere
        if provider_hint.startswith('command'):
            logger.info(f"Cohere detected by model name - model: {provider_hint}")
            return LLMProvider.COHERE
        
        # Fireworks path
        if provider_hint.startswith('accounts/fireworks'):
            logger.info(f"Fireworks detected by path - path: {provider_hint}")
            return LLMProvider.FIREWORKS
        
        # Mistral series -> MistralAI
        if provider_hint.startswith('mistral'):
            logger.info(f"MistralAI detected by model name - model: {provider_hint}")
            return LLMProvider.MISTRALAI
        
        # DeepSeek series -> DeepSeek
        if provider_hint.startswith('deepseek'):
            logger.info(f"DeepSeek detected by model name - model: {provider_hint}")
            return LLMProvider.DEEPSEEK
        
        # Grok series -> XAI
        if provider_hint.startswith('grok'):
            logger.info(f"XAI detected by model name - model: {provider_hint}")
            return LLMProvider.XAI
        
        # Sonar series -> Perplexity
        if provider_hint.startswith('sonar'):
            logger.info(f"Perplexity detected by model name - model: {provider_hint}")
            return LLMProvider.PERPLEXITY
        
        # 3. Fallback: Match by provider name keyword (order is important!)
        # More specific keywords should come first
        keyword_mapping = [
            # Match more specific keywords first
            ('google_genai', LLMProvider.GOOGLE_GENAI),
            ('google_vertexai', LLMProvider.GOOGLE_VERTEXAI),
            ('azure_openai', LLMProvider.AZURE_OPENAI),
            ('azure_ai', LLMProvider.AZURE_AI),
            ('bedrock_converse', LLMProvider.BEDROCK_CONVERSE),
            
            # Then match general keywords
            ('google', LLMProvider.GOOGLE_VERTEXAI),  # Google defaults to VertexAI
            ('vertex', LLMProvider.GOOGLE_VERTEXAI),
            ('openai', LLMProvider.OPENAI),
            ('anthropic', LLMProvider.ANTHROPIC),
            ('huggingface', LLMProvider.HUGGINGFACE),
            ('hf', LLMProvider.HUGGINGFACE),
            ('ollama', LLMProvider.OLLAMA),
            ('groq', LLMProvider.GROQ),
            ('cohere', LLMProvider.COHERE),
            ('fireworks', LLMProvider.FIREWORKS),
            ('together', LLMProvider.TOGETHER),
            ('mistralai', LLMProvider.MISTRALAI),
            ('bedrock', LLMProvider.BEDROCK),
            ('ibm', LLMProvider.IBM),
            ('nvidia', LLMProvider.NVIDIA),
            ('xai', LLMProvider.XAI),
            ('perplexity', LLMProvider.PERPLEXITY),
        ]
        
        for keyword, provider in keyword_mapping:
            if keyword in provider_hint:
                logger.info(f"Provider detected by keyword - keyword: {keyword}, provider: {provider.value}")
                return provider
        
        logger.warning(f"Could not detect provider, returning UNKNOWN - hint: {provider_hint}")
        return LLMProvider.UNKNOWN
    
    def validate_and_fix_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validates and fixes the message sequence to meet the requirements of a specific provider.
        
        Args:
            messages: The original list of messages.
            
        Returns:
            A tuple containing the fixed message list and a record of the fixes made.
        """
        logger.info(f"Starting message validation and fixing - Provider: {self.provider.value}, Original message count: {len(messages)}")
        
        if not messages:
            logger.warning("Message list is empty")
            return [], ["Message list is empty"]
        
        fixed_messages = [msg.copy() for msg in messages]
        fixes = []
        
        # 1. Basic validation and cleanup
        logger.debug("Performing basic validation and cleanup")
        fixed_messages, basic_fixes = self._basic_cleanup(fixed_messages)
        fixes.extend(basic_fixes)
        
        # 2. Handle system message position
        if not self.requirements.supports_system_anywhere:
            logger.debug("Handling system message position")
            fixed_messages, system_fixes = self._fix_system_message_position(fixed_messages)
            fixes.extend(system_fixes)
        
        # 3. Ensure starts with user message (if required)
        if self.requirements.must_start_with_user:
            logger.debug("Ensuring the sequence starts with a user message")
            fixed_messages, start_fixes = self._ensure_starts_with_user(fixed_messages)
            fixes.extend(start_fixes)
        
        # 4. Handle consecutive same-role messages
        if not self.requirements.allow_consecutive_same_role:
            logger.debug("Handling consecutive same-role messages")
            fixed_messages, consecutive_fixes = self._fix_consecutive_messages(fixed_messages)
            fixes.extend(consecutive_fixes)
        
        # 5. Handle alternating requirement
        if self.requirements.requires_alternating:
            logger.debug("Enforcing alternating pattern requirement")
            fixed_messages, alternating_fixes = self._enforce_alternating_pattern(fixed_messages)
            fixes.extend(alternating_fixes)
        
        # 6. Handle tool call messages
        if self.requirements.supports_tool_calls:
            logger.debug("Handling tool call messages")
            fixed_messages, tool_fixes = self._fix_tool_messages(fixed_messages)
            fixes.extend(tool_fixes)
        
        # 7. Fallback: Ensure no empty content
        logger.debug("Ensuring no empty content")
        fixed_messages, no_empty_fixes = self._ensure_no_empty_content(fixed_messages)
        fixes.extend(no_empty_fixes)
        
        # 8. Final validation
        logger.debug("Performing final validation")
        final_issues = self._final_validation(fixed_messages)
        if final_issues:
            fixes.extend([f"Final validation found issue: {issue}" for issue in final_issues])
        
        logger.info(f"Message validation and fixing complete - Fixed message count: {len(fixed_messages)}, Fix operations count: {len(fixes)}")
        return fixed_messages, fixes
    
    def _basic_cleanup(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Basic cleanup: remove empty messages, standardize role names, etc."""
        logger.debug(f"Starting basic cleanup - Input message count: {len(messages)}")
        
        fixes = []
        cleaned_messages = []
        
        for i, msg in enumerate(messages):
            # Check for necessary fields
            if 'role' not in msg:
                fixes.append(f"Message {i} is missing 'role' field, skipping")
                logger.warning(f"Message {i} is missing 'role' field")
                continue
            
            # Standardize role name
            role = msg['role'].lower().strip()
            original_role = role
            if role not in ['user', 'assistant', 'system', 'tool']:
                if role in ['human', 'user_message']:
                    role = 'user'
                elif role in ['ai', 'assistant_message', 'model']:
                    role = 'assistant'
                else:
                    fixes.append(f"Message {i} has an unrecognized role '{msg['role']}', setting to 'user'")
                    logger.warning(f"Message {i} has an unrecognized role '{msg['role']}'")
                    role = 'user'
                
                if original_role != role:
                    logger.debug(f"Role standardized - Message {i}: {original_role} -> {role}")
            
            # Check content and tool calls
            content = msg.get('content', '')
            if content is None:
                content = ''
            else:
                content = str(content).strip()
            
            has_tool_calls = 'tool_calls' in msg and msg['tool_calls']
            is_tool_message = role == 'tool'
            
            # For messages with empty content, perform stricter checks
            if not content:
                if has_tool_calls:
                    # Assistant message with tool calls but no content is normal
                    logger.debug(f"Message {i} is an assistant message with tool calls but no content")
                    pass
                elif is_tool_message:
                    # Tool message must have content
                    fixes.append(f"Message {i} is a tool message but content is empty, skipping")
                    logger.warning(f"Message {i} is a tool message but content is empty")
                    continue
                else:
                    # Skip other empty messages
                    fixes.append(f"Message {i} has empty content and no tool calls, skipping")
                    logger.debug(f"Message {i} has empty content and no tool calls")
                    continue
            
            cleaned_msg = {
                'role': role,
                'content': content
            }
            
            # Preserve other important fields
            for key in ['tool_calls', 'tool_call_id', 'name']:
                if key in msg and msg[key] is not None:
                    cleaned_msg[key] = msg[key]
            
            cleaned_messages.append(cleaned_msg)
        
        logger.debug(f"Basic cleanup complete - Output message count: {len(cleaned_messages)}, Fix count: {len(fixes)}")
        return cleaned_messages, fixes
    
    def _fix_system_message_position(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Moves the system message to the beginning (if required by the provider)."""
        logger.debug(f"Fixing system message position - Input message count: {len(messages)}")
        
        fixes = []
        system_messages = []
        other_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_messages.append(msg)
            else:
                other_messages.append(msg)
        
        logger.debug(f"System message statistics - System messages: {len(system_messages)}, Other messages: {len(other_messages)}")
        
        if system_messages and other_messages:
            # Merge system messages
            if len(system_messages) > 1:
                combined_content = '\n\n'.join(msg['content'] for msg in system_messages if msg['content'])
                system_messages = [{'role': 'system', 'content': combined_content}]
                fixes.append(f"Merged {len(system_messages)} system messages")
                logger.info(f"Merged {len(system_messages)} system messages")
            
            fixed_messages = system_messages + other_messages
            fixes.append("Moved system message to the beginning")
            logger.info("Moved system message to the beginning")
            return fixed_messages, fixes
        
        return messages, fixes
    
    def _ensure_starts_with_user(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Ensures the message sequence starts with a user message."""
        logger.debug(f"Ensuring starts with user message - Input message count: {len(messages)}")
        
        fixes = []
        
        if not messages:
            return messages, fixes
        
        # Find the first non-system message
        first_non_system_idx = 0
        for i, msg in enumerate(messages):
            if msg['role'] != 'system':
                first_non_system_idx = i
                break
        
        logger.debug(f"Position of the first non-system message: {first_non_system_idx}")
        
        if first_non_system_idx < len(messages):
            first_msg = messages[first_non_system_idx]
            if first_msg['role'] != 'user':
                # Insert a default user message
                default_user_msg = {
                    'role': 'user', 
                    'content': 'Please continue our conversation.'
                }
                messages.insert(first_non_system_idx, default_user_msg)
                fixes.append("Inserted a user message at the beginning to meet provider requirements")
                logger.info("Inserted a user message at the beginning")
        
        return messages, fixes
    
    def _fix_consecutive_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Fixes consecutive messages of the same role."""
        logger.debug(f"Fixing consecutive messages - Input message count: {len(messages)}")
        
        fixes = []
        fixed_messages = []
        
        i = 0
        while i < len(messages):
            current_msg = messages[i]
            
            # Find consecutive messages of the same role
            consecutive_msgs = [current_msg]
            j = i + 1
            while j < len(messages) and messages[j]['role'] == current_msg['role']:
                consecutive_msgs.append(messages[j])
                j += 1
            
            # If there are consecutive messages, merge them
            if len(consecutive_msgs) > 1:
                logger.debug(f"Found consecutive messages - Role: {current_msg['role']}, Count: {len(consecutive_msgs)}")
                
                # Collect all non-empty content
                content_parts = []
                for msg in consecutive_msgs:
                    content = msg.get('content', '').strip()
                    if content:
                        content_parts.append(content)
                
                # Merge content - ensure it's not empty
                if content_parts:
                    combined_content = '\n\n'.join(content_parts)
                else:
                    # If all messages have no content, check for tool calls
                    has_tool_calls = any(msg.get('tool_calls') for msg in consecutive_msgs)
                    if has_tool_calls and current_msg['role'] == 'assistant':
                        # Assistant messages with tool calls but no content are normal
                        combined_content = ''
                        logger.debug("Merged assistant message has tool calls but no content")
                    else:
                        # Provide appropriate default content based on role
                        if current_msg['role'] == 'user':
                            combined_content = 'Please continue.'
                        elif current_msg['role'] == 'assistant':
                            combined_content = 'Continue.'
                        elif current_msg['role'] == 'system':
                            combined_content = 'System prompt.'
                        elif current_msg['role'] == 'tool':
                            combined_content = 'Tool execution completed.'
                        else:
                            combined_content = 'Continue.'
                        logger.debug(f"Provided default content for merged message - Role: {current_msg['role']}")
                
                # Create the merged message
                merged_msg = {
                    'role': current_msg['role'],
                    'content': combined_content
                }
                
                # Merge special fields like tool calls
                all_tool_calls = []
                for msg in consecutive_msgs:
                    if msg.get('tool_calls'):
                        all_tool_calls.extend(msg['tool_calls'])
                
                if all_tool_calls:
                    merged_msg['tool_calls'] = all_tool_calls
                    logger.debug(f"Merged tool calls - Count: {len(all_tool_calls)}")
                
                # Preserve other important fields (prioritize fields from the last message)
                for msg in consecutive_msgs[1:]:
                    for key in ['tool_call_id', 'name']:
                        if key in msg and msg[key] is not None:
                            merged_msg[key] = msg[key]
                
                fixed_messages.append(merged_msg)
                fixes.append(f"Merged {len(consecutive_msgs)} consecutive {current_msg['role']} messages")
                logger.info(f"Merged {len(consecutive_msgs)} consecutive {current_msg['role']} messages")
            else:
                # Single message, add directly (ensure content is not empty)
                msg_copy = current_msg.copy()
                
                # For Gemini, ensure all messages have non-empty content
                if self.provider in (LLMProvider.GOOGLE_GENAI, LLMProvider.GOOGLE_VERTEXAI):
                    if 'content' not in msg_copy or msg_copy['content'] is None:
                        msg_copy['content'] = ''
                    
                    # If content is empty and there are no tool calls, provide default content
                    if not str(msg_copy['content']).strip():
                        has_tool_calls = msg_copy.get('tool_calls')
                        if not (has_tool_calls and msg_copy['role'] == 'assistant'):
                            if msg_copy['role'] == 'user':
                                msg_copy['content'] = 'Please continue.'
                            elif msg_copy['role'] == 'assistant':
                                msg_copy['content'] = 'Continue.'
                            elif msg_copy['role'] == 'system':
                                msg_copy['content'] = 'System prompt.'
                            elif msg_copy['role'] == 'tool':
                                msg_copy['content'] = 'Tool execution completed.'
                            fixes.append(f"Added default content to empty message {i}")
                            logger.debug(f"Added default content to empty message {i}")
                
                fixed_messages.append(msg_copy)
            
            i = j
        
        logger.debug(f"Consecutive message fixing complete - Fixed message count: {len(fixed_messages)}")
        return fixed_messages, fixes
    
    def _enforce_alternating_pattern(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Enforces a user-assistant alternating pattern."""
        logger.debug(f"Enforcing alternating pattern - Input message count: {len(messages)}")
        
        fixes = []
        fixed_messages = []
        
        for i, msg in enumerate(messages):
            role = msg['role']
            
            # System and tool messages are preserved directly
            if role in ['system', 'tool']:
                # Ensure system and tool messages also have content (for Gemini)
                msg_copy = msg.copy()
                if self.provider in (LLMProvider.GOOGLE_GENAI, LLMProvider.GOOGLE_VERTEXAI):
                    if 'content' not in msg_copy or not str(msg_copy.get('content', '')).strip():
                        if role == 'system':
                            msg_copy['content'] = 'System prompt.'
                        elif role == 'tool':
                            msg_copy['content'] = 'Tool execution completed.'
                        fixes.append(f"Added default content for {role} message")
                        logger.debug(f"Added default content for {role} message")
                fixed_messages.append(msg_copy)
                continue
            
            # Check if an alternating message needs to be inserted
            if fixed_messages:
                last_non_system_msg = None
                for prev_msg in reversed(fixed_messages):
                    if prev_msg['role'] not in ['system', 'tool']:
                        last_non_system_msg = prev_msg
                        break
                
                if last_non_system_msg and last_non_system_msg['role'] == role:
                    # An opposite role message needs to be inserted
                    opposite_role = 'assistant' if role == 'user' else 'user'
                    
                    # Provide appropriate default content based on role
                    if opposite_role == 'assistant':
                        placeholder_content = 'Continue.'
                    else:
                        placeholder_content = 'Please continue.'
                    
                    placeholder_msg = {
                        'role': opposite_role,
                        'content': placeholder_content
                    }
                    fixed_messages.append(placeholder_msg)
                    fixes.append(f"Inserted {opposite_role} message to maintain alternating pattern")
                    logger.info(f"Inserted {opposite_role} message to maintain alternating pattern")
            
            # Add the current message, ensuring content is not empty
            msg_copy = msg.copy()
            if self.provider in (LLMProvider.GOOGLE_GENAI, LLMProvider.GOOGLE_VERTEXAI):
                if 'content' not in msg_copy or not str(msg_copy.get('content', '')).strip():
                    # Empty messages without tool calls need default content
                    has_tool_calls = msg_copy.get('tool_calls')
                    if not (has_tool_calls and role == 'assistant'):
                        if role == 'user':
                            msg_copy['content'] = 'Please continue.'
                        elif role == 'assistant':
                            msg_copy['content'] = 'Continue.'
                        fixes.append(f"Added default content to empty {role} message")
                        logger.debug(f"Added default content to empty {role} message")
            
            fixed_messages.append(msg_copy)
        
        logger.debug(f"Alternating pattern enforcement complete - Fixed message count: {len(fixed_messages)}")
        return fixed_messages, fixes
    
    def _fix_tool_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Fixes messages related to tool calls."""
        logger.debug(f"Fixing tool call messages - Input message count: {len(messages)}, Provider: {self.provider.value}")
        
        fixes = []
        fixed_messages = []
        
        for i, msg in enumerate(messages):
            msg_copy = msg.copy()
            
            # Special handling for Gemini's empty content issue
            if self.provider in (LLMProvider.GOOGLE_GENAI, LLMProvider.GOOGLE_VERTEXAI):
                # Ensure all messages have a content field
                if 'content' not in msg_copy:
                    msg_copy['content'] = ''
                elif msg_copy['content'] is None:
                    msg_copy['content'] = ''
                else:
                    # Ensure content is a string
                    msg_copy['content'] = str(msg_copy['content'])
                
                # Check if content is empty
                content_is_empty = not msg_copy['content'].strip()
                has_tool_calls = bool(msg_copy.get('tool_calls'))
                role = msg_copy['role']
                
                # Decide how to handle empty content based on role and tool calls
                if content_is_empty:
                    # Google Gemini requires non-empty content for all messages, even for assistants with tool_calls
                    if role == 'assistant' and has_tool_calls:
                        msg_copy['content'] = 'Calling tools.'
                        fixes.append(f"Added default content to message {i} ({role}) to avoid empty message (tool call placeholder)")
                        logger.debug(f"Added tool call placeholder content to message {i} ({role})")
                    else:
                        # Provide default content in other cases
                        if role == 'user':
                            msg_copy['content'] = 'Please continue.'
                        elif role == 'assistant':
                            msg_copy['content'] = 'Continue.'
                        elif role == 'system':
                            msg_copy['content'] = 'System prompt.'
                        elif role == 'tool':
                            msg_copy['content'] = 'Tool execution completed.'
                        else:
                            msg_copy['content'] = 'Continue.'
                        fixes.append(f"Added default content to message {i} ({role}) to avoid empty message")
                        logger.debug(f"Added default content to message {i} ({role})")
            
            fixed_messages.append(msg_copy)
        
        logger.debug(f"Tool call message fixing complete - Fixed message count: {len(fixed_messages)}")
        return fixed_messages, fixes
    
    def _ensure_no_empty_content(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Ensures all message content is non-empty (final fallback)."""
        logger.debug(f"Final fallback check for empty content - Input message count: {len(messages)}")
        
        fixes = []
        for i, msg in enumerate(messages):
            if 'content' not in msg or not str(msg.get('content', '')).strip():
                default = 'Continue.' if msg['role'] == 'assistant' else 'Please continue.'
                if msg['role'] == 'system':
                    default = 'System prompt.'
                elif msg['role'] == 'tool':
                    default = 'Tool execution completed.'
                msg['content'] = default
                fixes.append(f"Final fallback: Filled in default content for message {i} ({msg['role']})")
                logger.debug(f"Final fallback: Filled in default content for message {i} ({msg['role']})")
        
        logger.debug(f"Empty content check complete - Fix count: {len(fixes)}")
        return messages, fixes
    
    def _final_validation(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Final validation of the message sequence."""
        logger.debug(f"Final validation of message sequence - Input message count: {len(messages)}")
        
        issues = []
        
        if not messages:
            issues.append("Message list is empty")
            logger.warning("Final validation: Message list is empty")
            return issues
        
        # Check if provider requirements are met
        if self.requirements.must_start_with_user:
            first_non_system = None
            for msg in messages:
                if msg['role'] != 'system':
                    first_non_system = msg
                    break
            
            if first_non_system and first_non_system['role'] != 'user':
                issues.append("Message sequence does not start with a user message")
                logger.warning("Final validation: Message sequence does not start with a user message")
        
        # Check for consecutive messages
        if not self.requirements.allow_consecutive_same_role:
            prev_role = None
            for i, msg in enumerate(messages):
                if msg['role'] not in ['system', 'tool']:
                    if prev_role == msg['role']:
                        issues.append(f"Found consecutive {msg['role']} messages at position {i}")
                        logger.warning(f"Final validation: Found consecutive {msg['role']} messages at position {i}")
                    prev_role = msg['role']
        
        logger.debug(f"Final validation complete - Issue count: {len(issues)}")
        return issues
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Gets information about the current provider."""
        info = {
            'provider': self.provider.value,
            'requirements': {
                'must_start_with_user': self.requirements.must_start_with_user,
                'allow_consecutive_same_role': self.requirements.allow_consecutive_same_role,
                'requires_alternating': self.requirements.requires_alternating,
                'supports_system_anywhere': self.requirements.supports_system_anywhere,
                'supports_tool_calls': self.requirements.supports_tool_calls,
                'max_consecutive_assistant': self.requirements.max_consecutive_assistant,
                'requires_user_after_tool': self.requirements.requires_user_after_tool
            }
        }
        logger.debug(f"Getting provider info - Provider: {self.provider.value}")
        return info

# For backward compatibility, keep the old function name
def fix_gemini_message_sequence(messages: List[Dict[str, Any]], 
                               provider: str = "google") -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Backward compatibility function - now uses the universal message manager.
    
    Args:
        messages: The list of messages.
        provider: The LLM provider.
        
    Returns:
        A tuple containing the fixed message list and a record of the fixes made.
    """
    logger.info(f"Backward compatibility call - Message count: {len(messages)}, Provider: {provider}")
    manager = UniversalMessageManager(provider)
    return manager.validate_and_fix_messages(messages)

# Main public interface
def validate_and_fix_messages(messages: List[Dict[str, Any]], 
                            provider: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Universal message validation and fixing function.
    
    Args:
        messages: The list of messages.
        provider: The LLM provider (optional, uses default compatibility mode if not provided).
        
    Returns:
        A tuple containing the fixed message list and a record of the fixes made.
    """
    logger.info(f"Universal interface call - Message count: {len(messages)}, Provider: {provider or 'auto-detect'}")
    manager = UniversalMessageManager(provider)
    return manager.validate_and_fix_messages(messages)