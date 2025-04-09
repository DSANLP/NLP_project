import os
import platform
import signal

# 如果是Windows平台，简单处理SIGINT
if platform.system() == 'Windows':
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
    signal.signal(signal.SIGINT, lambda x, y: None) 