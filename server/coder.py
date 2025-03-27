from aider.models import Model
from aider.coders import Coder as BaseCoder, ArchitectCoder

from server.utils import ChatChunkData, logger

# patch Coder
def should_stop(self):
    return self.stop_event.is_set()

def on_data_update(self, fn):
    self.on_data_update = fn

def emit_data_update(self, data):
    if self.on_data_update:
        self.on_data_update(data)
        

BaseCoder.should_stop = should_stop
BaseCoder.emit_data_update = emit_data_update
BaseCoder.on_data_update = on_data_update

# Create patched Coder subclass
class Coder(BaseCoder):
    def run_stream(self, *args, **kwargs):
        for chunk in super().run_stream(*args, **kwargs):
            if self.stop or self.should_stop():
                logger.info(f'coder stream intercept')
                raise KeyboardInterrupt("Generation stopped by user")
            yield chunk
    def stop_stream(self):
        self.stop = True
        logger.info('Coder stream stopped by user')

# patch ArchitectCoder
original_reply_completed = ArchitectCoder.reply_completed
def reply_completed(self):
    self.emit_data_update(ChatChunkData(event='editor-start'))
    result = original_reply_completed(self)
    self.emit_data_update(ChatChunkData(event='editor-end'))
    return result


ArchitectCoder.reply_completed = reply_completed
