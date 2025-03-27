# Standard library imports
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from queue import Queue
from threading import Event, Thread
from typing import Any, Dict, Iterator, List, Literal, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ != '__main__':
    # Remove current working directory from Python path to avoid module import conflicts
    # This prevents local files from shadowing installed packages with the same name
    cwd = os.getcwd()
    if cwd in sys.path:
        sys.path.remove(cwd)


# Third-party imports
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from aider.io import InputOutput
from aider.models import Model

# Local imports
from server.coder import ArchitectCoder, Coder
from server.utils import (
    ChatChunkData,
    ChatModeType,
    ChatSessionData,
    ChatSessionReference,
    ChatSetting,
    ModelSetting,
    logger,
    provider_env_map,
)

class CaptureIO(InputOutput):
    lines: List[str]
    error_lines: List[str]
    write_files: Dict[str, str]

    def __init__(self, *args, **kwargs):
        self.lines = []
        # when spawned in node process, tool_error will be called
        # so we need to create before super().__init__
        self.error_lines = []
        self.write_files = {}
        super().__init__(*args, **kwargs)

    def tool_output(self, msg="", log_only=False, bold=False):
        logger.info(f'aider tool: {msg}')
        if not log_only:
            self.lines.append(msg)
        super().tool_output(msg, log_only=log_only, bold=bold)

    def tool_error(self, msg):
        logger.error(f'aider tool: {msg}')
        self.error_lines.append(msg)
        super().tool_error(msg)

    def tool_warning(self, msg):
        logger.warning(f'aider tool: {msg}')
        self.lines.append(msg)
        super().tool_warning(msg)

    def get_captured_lines(self):
        lines = self.lines
        self.lines = []
        return lines
    
    def get_captured_error_lines(self):
        lines = self.error_lines
        self.error_lines = []
        return lines

    def write_text(self, filename, content):
        logger.info(f'write {filename}')
        self.write_files[filename] = content
    
    def read_text(self, filename):
        logger.info(f'read {filename}')
        if filename in self.write_files:
            return self.write_files[filename]
        return super().read_text(filename)

    def get_captured_write_files(self):
        write_files = self.write_files
        self.write_files = {}
        return write_files
    
    def confirm_ask(
        self,
        question: str,
        default="y",
        subject=None,
        explicit_yes_required=False,
        group=None,
        allow_never=False,
    ):
        logger.info(f'confirm_ask: {question}, subject: {subject}, group: {group}')
        # create new file
        if 'Create new file' in question:
            return True
        elif 'Edit the files' in question:
            return True
        return False

class ChatSessionManager:
    """Manages chat sessions with the AI coder, handling message streaming, model updates, and file operations."""
    
    chat_type: ChatModeType
    diff_format: str
    reference_list: List[ChatSessionReference]
    setting: Optional[ChatSetting] = None
    confirm_ask_result: Optional[Any] = None

    coder: Coder
    is_chat_streaming: bool

    def __init__(self):
        """Initialize the chat session manager with default settings and coder instance."""
        self.stop_event = Event()
        self.is_chat_streaming = False
        model = Model('gpt-4o')
        io = CaptureIO(
            pretty=False,
            yes=False,
            dry_run=False,
            encoding='utf-8',
            fancy_input=False,
        )
        self.io = io

        coder = Coder.create(
            main_model=model,
            io=io,
            edit_format='ask',
            use_git=False,
        )
        coder.yield_stream = True
        coder.stream = True
        coder.pretty = False
        self.coder = coder
        self._update_patch_coder()

        self.chat_type = 'ask'
        self.diff_format = 'diff'
        self.reference_list = []

        self.confirm_ask_event = Event()
        self.queue = Queue()
    
    def _update_patch_coder(self):
        """Update the coder instance with the latest data update callback."""
        self.coder.on_data_update(lambda data: self.queue.put(data))

    def update_model(self, setting: ChatSetting):
        """Update the AI model configuration with new settings.
        
        Args:
            setting: ChatSetting object containing model configuration
        """
        if self.setting != setting:
            self.setting = setting
            model = Model(setting.main_model.model)
            
            # Configure main model environment
            self._configure_model_env(setting.main_model)
 
            # Configure editor model if provided
            if setting.editor_model:
                model.editor_model = Model(setting.editor_model.model)
                self._configure_model_env(setting.editor_model)
            
            self.coder = Coder.create(from_coder=self.coder, main_model=model)
    
    def _configure_model_env(self, setting: ModelSetting):
        """Configure environment variables for the AI model based on provider settings.
        
        Args:
            setting: ModelSetting object containing provider-specific configuration
        """
        # update os env
        config = provider_env_map[setting.provider]
        if isinstance(config, str):
            os.environ[config] = setting.api_key
        # explicitly handle configs that need multiple env variables, like base urls and api keys
        elif isinstance(config, dict):
            for key, value in config.items():
                os.environ[value] = getattr(setting, key)
    
    def update_coder(self):
        """Update the coder instance with current chat type, diff format, and file references."""
        self.stop_event.clear()
        self.coder = Coder.create(
            from_coder=self.coder,
            edit_format=self.diff_format if self.chat_type == 'code' else self.chat_type,
            fnames=(item.fs_path for item in self.reference_list if not item.readonly),
            read_only_fnames=(item.fs_path for item in self.reference_list if item.readonly),
        )
        if self.chat_type == 'architect':
            self.coder.main_model.editor_edit_format = self.diff_format

        self._update_patch_coder()

    def chat(self, data: ChatSessionData) -> Iterator[ChatChunkData]:
        """Handle a chat session with the AI coder, streaming responses back.
        
        Args:
            data: ChatSessionData object containing chat configuration and message
            
        Returns:
            Iterator yielding ChatChunkData events for streaming responses
        """
        need_update_coder = False
        data.reference_list.sort(key=lambda x: x.fs_path)

        if data.chat_type != self.chat_type or data.diff_format != self.diff_format:
            need_update_coder = True
            self.chat_type = data.chat_type
            self.diff_format = data.diff_format
        if data.reference_list != self.reference_list:
            need_update_coder = True
            self.reference_list = data.reference_list

        if need_update_coder:
            self.update_coder()
        
        # Start coder thread
        thread = Thread(target=self._coder_thread, args=(data.message,))
        thread.start()

        # Yield data from queue
        while True:
            chunk = self.queue.get()
            yield chunk
            if chunk.event == 'end':
                break

    def _coder_thread(self, message: str):
        """Run the coder in a separate thread to process the message and generate responses.
        
        Args:
            message: The user's message to process
        """
        try:
            self.coder.init_before_message()
            while message and not self.stop_event.is_set():
                self.coder.reflected_message = None
                for msg in self.coder.run_stream(message):
                    if self.stop_event.is_set():
                        break
                    data = {
                        "chunk": msg,
                    }
                    self.queue.put(ChatChunkData(event='data', data=data))

                if not self.stop_event.is_set() and self.coder.usage_report:
                    data = { "usage": self.coder.usage_report }
                    self.queue.put(ChatChunkData(event='usage', data=data))
                
                if not self.coder.reflected_message:
                    break

                if self.coder.num_reflections >= self.coder.max_reflections:
                    self.coder.io.tool_warning(f"Only {self.coder.max_reflections} reflections allowed, stopping.")
                    break

                self.coder.num_reflections += 1
                message = self.coder.reflected_message

                self.queue.put(ChatChunkData(event='reflected', data={"message": message}))

                error_lines = self.coder.io.get_captured_error_lines()
                if error_lines:
                    if not message:
                        raise Exception('\n'.join(error_lines))
                    else:
                        self.queue.put(ChatChunkData(event='log', data={"message": '\n'.join(error_lines)}))

            # get write files
            write_files = self.io.get_captured_write_files()
            if write_files:
                data = {
                    "write": write_files,
                }
                self.queue.put(ChatChunkData(event='write', data=data))

        except Exception as e:
            # send error to client
            error_data = {
                "error": str(e)
            }
            self.queue.put(ChatChunkData(event='error', data=error_data))
        finally:
            # send end event to client
            self.queue.put(ChatChunkData(event='end'))

    
    def confirm_ask(self):
        """Wait for user confirmation on file operations."""
        self.confirm_ask_event.clear()
        self.confirm_ask_event.wait()

    def confirm_ask_reply(self):
        """Signal that user has replied to confirmation prompt."""
        self.confirm_ask_event.set()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ChatSessionManager()

@app.post('/api/chat')
async def sse(request: Request):
    data = await request.json()
    data['reference_list'] = [ChatSessionReference(**item) for item in data['reference_list']]

    chat_session_data = ChatSessionData(**data)

    async def event_generator():
        try:
            for msg in manager.chat(chat_session_data):
                # If client closed the connection
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping chat session")
                    manager.coder.stop_stream()
                    break
                if msg.data:
                    yield {
                        "event": msg.event,
                        "data": json.dumps(msg.data)
                    }
                else:
                    yield {
                        "event": msg.event,
                        "data": ""
                    }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())

@app.delete('/api/chat')
async def clear():
    manager.coder.done_messages = []
    manager.coder.cur_messages = []
    return JSONResponse(content={})

@app.put('/api/chat/session')
async def set_history(request: Request):
    data = await request.json()
    manager.coder.done_messages = data
    manager.coder.cur_messages = []
    return JSONResponse(content={})

@app.post('/api/chat/setting')
async def update_setting(request: Request):
    data = await request.json()
    # Create ModelSetting instances for both main and editor models
    data['main_model'] = ModelSetting(**data['main_model'])
    if 'editor_model' in data and data['editor_model']:
        data['editor_model'] = ModelSetting(**data['editor_model'])
    setting = ChatSetting(**data)

    manager.update_model(setting)
    return JSONResponse(content={})

@app.post('/api/chat/confirm/ask')
async def confirm_ask():
    manager.confirm_ask()
    return JSONResponse(content=manager.confirm_ask_result)

@app.post('/api/chat/confirm/reply')
async def confirm_reply(request: Request):
    data = await request.json()
    manager.confirm_ask_result = data
    manager.confirm_ask_reply()
    return JSONResponse(content={})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=5000, timeout_keep_alive=600)
