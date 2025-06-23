"""
通用消息管理器 - 支持所有主流LLM提供商
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """支持的LLM提供商 - 与LangChain init_chat_model完全兼容"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE_GENAI = "google_genai"  # Google GenAI
    GOOGLE_VERTEXAI = "google_vertexai"  # Google VertexAI
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    # 新增的提供商
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
    """LLM提供商的消息序列要求"""
    must_start_with_user: bool = False
    allow_consecutive_same_role: bool = True
    requires_alternating: bool = False
    supports_system_anywhere: bool = True
    supports_tool_calls: bool = True
    max_consecutive_assistant: int = 999
    requires_user_after_tool: bool = False

class UniversalMessageManager:
    """通用消息管理器，支持所有主流LLM提供商"""
    
    # 各提供商的消息序列要求
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
        初始化消息管理器
        
        Args:
            provider: LLM提供商名称，如果为None则自动检测
        """
        self.provider = self._detect_provider(provider) if provider else LLMProvider.UNKNOWN
        self.requirements = self.PROVIDER_REQUIREMENTS[self.provider]
        
    def _detect_provider(self, provider_hint: str) -> LLMProvider:
        """
        根据提示检测LLM提供商
        
        遵循LangChain init_chat_model的推断规则：
        https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
        """
        if not provider_hint:
            return LLMProvider.UNKNOWN
            
        provider_hint = provider_hint.lower()
        
        # 1. 首先检查明确的提供商前缀（如 "openai:gpt-4"）
        if ':' in provider_hint:
            provider_prefix = provider_hint.split(':')[0]
            # 检查LangChain支持的提供商前缀
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
                return provider_mapping[provider_prefix]
        
        # 2. 按照LangChain的模型名称推断规则
        # GPT系列 -> OpenAI
        if any(provider_hint.startswith(prefix) for prefix in ['gpt-3', 'gpt-4', 'o1', 'o3']):
            return LLMProvider.OPENAI
        
        # Claude系列 -> Anthropic
        if provider_hint.startswith('claude'):
            return LLMProvider.ANTHROPIC
        
        # Amazon系列 -> Bedrock
        if provider_hint.startswith('amazon'):
            return LLMProvider.BEDROCK
        
        # Gemini系列 -> Google GenAI
        if provider_hint.startswith('gemini'):
            return LLMProvider.GOOGLE_GENAI
        
        # Command系列 -> Cohere
        if provider_hint.startswith('command'):
            return LLMProvider.COHERE
        
        # Fireworks路径
        if provider_hint.startswith('accounts/fireworks'):
            return LLMProvider.FIREWORKS
        
        # Mistral系列 -> MistralAI
        if provider_hint.startswith('mistral'):
            return LLMProvider.MISTRALAI
        
        # DeepSeek系列 -> DeepSeek
        if provider_hint.startswith('deepseek'):
            return LLMProvider.DEEPSEEK
        
        # Grok系列 -> XAI
        if provider_hint.startswith('grok'):
            return LLMProvider.XAI
        
        # Sonar系列 -> Perplexity
        if provider_hint.startswith('sonar'):
            return LLMProvider.PERPLEXITY
        
        # 3. 兜底：按提供商名称关键词匹配（顺序很重要！）
        # 更具体的关键词应该放在前面
        keyword_mapping = [
            # 先匹配更具体的关键词
            ('google_genai', LLMProvider.GOOGLE_GENAI),
            ('google_vertexai', LLMProvider.GOOGLE_VERTEXAI),
            ('azure_openai', LLMProvider.AZURE_OPENAI),
            ('azure_ai', LLMProvider.AZURE_AI),
            ('bedrock_converse', LLMProvider.BEDROCK_CONVERSE),
            
            # 然后匹配通用关键词
            ('google', LLMProvider.GOOGLE_VERTEXAI),  # Google默认VertexAI
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
                return provider
        
        return LLMProvider.UNKNOWN
    
    def validate_and_fix_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        验证并修复消息序列以符合特定提供商的要求
        
        Args:
            messages: 原始消息列表
            
        Returns:
            Tuple[修复后的消息列表, 修复操作记录]
        """
        if not messages:
            return [], ["消息列表为空"]
        
        fixed_messages = [msg.copy() for msg in messages]
        fixes = []
        
        # 1. 基础验证和清理
        fixed_messages, basic_fixes = self._basic_cleanup(fixed_messages)
        fixes.extend(basic_fixes)
        
        # 2. 处理系统消息位置
        if not self.requirements.supports_system_anywhere:
            fixed_messages, system_fixes = self._fix_system_message_position(fixed_messages)
            fixes.extend(system_fixes)
        
        # 3. 确保以用户消息开始（如果需要）
        if self.requirements.must_start_with_user:
            fixed_messages, start_fixes = self._ensure_starts_with_user(fixed_messages)
            fixes.extend(start_fixes)
        
        # 4. 处理连续同角色消息
        if not self.requirements.allow_consecutive_same_role:
            fixed_messages, consecutive_fixes = self._fix_consecutive_messages(fixed_messages)
            fixes.extend(consecutive_fixes)
        
        # 5. 处理交替要求
        if self.requirements.requires_alternating:
            fixed_messages, alternating_fixes = self._enforce_alternating_pattern(fixed_messages)
            fixes.extend(alternating_fixes)
        
        # 6. 处理工具调用消息
        if self.requirements.supports_tool_calls:
            fixed_messages, tool_fixes = self._fix_tool_messages(fixed_messages)
            fixes.extend(tool_fixes)
        
        # 7. 最终验证
        final_issues = self._final_validation(fixed_messages)
        if final_issues:
            fixes.extend([f"最终验证发现问题: {issue}" for issue in final_issues])
        
        return fixed_messages, fixes
    
    def _basic_cleanup(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """基础清理：移除空消息、标准化角色名称等"""
        fixes = []
        cleaned_messages = []
        
        for i, msg in enumerate(messages):
            # 检查必要字段
            if 'role' not in msg:
                fixes.append(f"消息 {i} 缺少 'role' 字段，跳过")
                continue
            
            # 标准化角色名称
            role = msg['role'].lower().strip()
            if role not in ['user', 'assistant', 'system', 'tool']:
                if role in ['human', 'user_message']:
                    role = 'user'
                elif role in ['ai', 'assistant_message', 'model']:
                    role = 'assistant'
                else:
                    fixes.append(f"消息 {i} 角色 '{msg['role']}' 无法识别，设为 'user'")
                    role = 'user'
            
            # 检查内容
            content = msg.get('content', '').strip() if msg.get('content') else ''
            if not content and 'tool_calls' not in msg and role != 'tool':
                fixes.append(f"消息 {i} 内容为空，跳过")
                continue
            
            cleaned_msg = {
                'role': role,
                'content': content
            }
            
            # 保留其他重要字段
            for key in ['tool_calls', 'tool_call_id', 'name']:
                if key in msg:
                    cleaned_msg[key] = msg[key]
            
            cleaned_messages.append(cleaned_msg)
        
        return cleaned_messages, fixes
    
    def _fix_system_message_position(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """将系统消息移动到开头（如果提供商要求）"""
        fixes = []
        system_messages = []
        other_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_messages.append(msg)
            else:
                other_messages.append(msg)
        
        if system_messages and other_messages:
            # 合并系统消息
            if len(system_messages) > 1:
                combined_content = '\n\n'.join(msg['content'] for msg in system_messages if msg['content'])
                system_messages = [{'role': 'system', 'content': combined_content}]
                fixes.append(f"合并了 {len(system_messages)} 个系统消息")
            
            fixed_messages = system_messages + other_messages
            fixes.append("将系统消息移动到开头")
            return fixed_messages, fixes
        
        return messages, fixes
    
    def _ensure_starts_with_user(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """确保消息序列以用户消息开始"""
        fixes = []
        
        if not messages:
            return messages, fixes
        
        # 找到第一个非系统消息
        first_non_system_idx = 0
        for i, msg in enumerate(messages):
            if msg['role'] != 'system':
                first_non_system_idx = i
                break
        
        if first_non_system_idx < len(messages):
            first_msg = messages[first_non_system_idx]
            if first_msg['role'] != 'user':
                # 插入一个默认用户消息
                default_user_msg = {
                    'role': 'user', 
                    'content': '请继续我们的对话。'
                }
                messages.insert(first_non_system_idx, default_user_msg)
                fixes.append("在开头插入用户消息以满足提供商要求")
        
        return messages, fixes
    
    def _fix_consecutive_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """修复连续的同角色消息"""
        fixes = []
        fixed_messages = []
        
        i = 0
        while i < len(messages):
            current_msg = messages[i]
            fixed_messages.append(current_msg)
            
            # 查找连续的同角色消息
            consecutive_msgs = [current_msg]
            j = i + 1
            while j < len(messages) and messages[j]['role'] == current_msg['role']:
                consecutive_msgs.append(messages[j])
                j += 1
            
            # 如果有连续消息，合并它们
            if len(consecutive_msgs) > 1:
                combined_content = '\n\n'.join(
                    msg['content'] for msg in consecutive_msgs 
                    if msg['content']
                )
                
                # 保留第一条消息，更新内容
                fixed_messages[-1]['content'] = combined_content
                
                # 保留工具调用等特殊字段
                for msg in consecutive_msgs[1:]:
                    for key in ['tool_calls', 'tool_call_id']:
                        if key in msg and key not in fixed_messages[-1]:
                            fixed_messages[-1][key] = msg[key]
                
                fixes.append(f"合并了 {len(consecutive_msgs)} 个连续的 {current_msg['role']} 消息")
            
            i = j
        
        return fixed_messages, fixes
    
    def _enforce_alternating_pattern(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """强制执行用户-助手交替模式"""
        fixes = []
        fixed_messages = []
        
        for i, msg in enumerate(messages):
            role = msg['role']
            
            # 系统和工具消息直接保留
            if role in ['system', 'tool']:
                fixed_messages.append(msg)
                continue
            
            # 检查是否需要插入交替消息
            if fixed_messages:
                last_non_system_msg = None
                for prev_msg in reversed(fixed_messages):
                    if prev_msg['role'] not in ['system', 'tool']:
                        last_non_system_msg = prev_msg
                        break
                
                if last_non_system_msg and last_non_system_msg['role'] == role:
                    # 需要插入相反角色的消息
                    opposite_role = 'assistant' if role == 'user' else 'user'
                    placeholder_msg = {
                        'role': opposite_role,
                        'content': '继续。' if opposite_role == 'assistant' else '请继续。'
                    }
                    fixed_messages.append(placeholder_msg)
                    fixes.append(f"插入 {opposite_role} 消息以维持交替模式")
            
            fixed_messages.append(msg)
        
        return fixed_messages, fixes
    
    def _fix_tool_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """修复工具调用相关的消息"""
        fixes = []
        
        # 这里可以添加特定于工具调用的修复逻辑
        # 例如确保工具消息有正确的 tool_call_id 等
        
        return messages, fixes
    
    def _final_validation(self, messages: List[Dict[str, Any]]) -> List[str]:
        """最终验证消息序列"""
        issues = []
        
        if not messages:
            issues.append("消息列表为空")
            return issues
        
        # 检查是否满足提供商要求
        if self.requirements.must_start_with_user:
            first_non_system = None
            for msg in messages:
                if msg['role'] != 'system':
                    first_non_system = msg
                    break
            
            if first_non_system and first_non_system['role'] != 'user':
                issues.append("消息序列未以用户消息开始")
        
        # 检查连续消息
        if not self.requirements.allow_consecutive_same_role:
            prev_role = None
            for i, msg in enumerate(messages):
                if msg['role'] not in ['system', 'tool']:
                    if prev_role == msg['role']:
                        issues.append(f"发现连续的 {msg['role']} 消息在位置 {i}")
                    prev_role = msg['role']
        
        return issues
    
    def get_provider_info(self) -> Dict[str, Any]:
        """获取当前提供商的信息"""
        return {
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

# 为了向后兼容，保留原有的函数名
def fix_gemini_message_sequence(messages: List[Dict[str, Any]], 
                               provider: str = "google") -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    向后兼容函数 - 现在使用通用消息管理器
    
    Args:
        messages: 消息列表
        provider: LLM提供商
        
    Returns:
        Tuple[修复后的消息列表, 修复操作记录]
    """
    manager = UniversalMessageManager(provider)
    return manager.validate_and_fix_messages(messages)

# 主要的公共接口
def validate_and_fix_messages(messages: List[Dict[str, Any]], 
                            provider: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    通用消息验证和修复函数
    
    Args:
        messages: 消息列表
        provider: LLM提供商（可选，如果不提供则使用默认兼容模式）
        
    Returns:
        Tuple[修复后的消息列表, 修复操作记录]
    """
    manager = UniversalMessageManager(provider)
    return manager.validate_and_fix_messages(messages)