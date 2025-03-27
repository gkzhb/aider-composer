from typing import Optional, List, Literal
from dataclasses import dataclass
import logging

@dataclass
class ChatChunkData:
    # event: data, usage, write, end, error, reflected, log, editor
    # data: yield chunk message
    # usage: yield usage report
    # write: yield write files
    # end: end of chat
    # error: yield error message
    # reflected: yield reflected message
    # log: yield log message
    # editor: editor start working
    event: str
    data: Optional[dict] = None
    
@dataclass
class ModelSetting:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str] = None

@dataclass
class ChatSetting:
    main_model: ModelSetting
    editor_model: Optional[ModelSetting] = None

provider_env_map = {
    'deepseek': 'DEEPSEEK_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'ollama': {
        'base_url': 'OLLAMA_API_BASE',
    },
    'openrouter': 'OPENROUTER_API_KEY',
    'openai_compatible': {
        'api_key': 'OPENAI_API_KEY',
        'base_url': 'OPENAI_API_BASE',
    },
    'gemini': 'GEMINI_API_KEY',
}

@dataclass
class ChatSessionReference:
    readonly: bool
    fs_path: str

@dataclass
class ChatSessionData:
    chat_type: str
    diff_format: str
    message: str
    reference_list: List[ChatSessionReference]

ChatModeType = Literal['ask', 'code', 'architect']


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)