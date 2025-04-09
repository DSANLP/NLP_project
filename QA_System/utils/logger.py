import os
import datetime
import sys

# 日志文件目录
LOG_DIR = "log"

# 日志级别
LOG_LEVEL_DEBUG = 10
LOG_LEVEL_INFO = 20
LOG_LEVEL_WARNING = 30
LOG_LEVEL_ERROR = 40
LOG_LEVEL_CRITICAL = 50

# 当前日志级别
CURRENT_LOG_LEVEL = LOG_LEVEL_INFO

# 日志颜色代码
COLORS = {
    "RESET": "\033[0m",
    "DEBUG": "\033[36m",    # 青色
    "INFO": "\033[32m",     # 绿色
    "WARNING": "\033[33m",  # 黄色
    "ERROR": "\033[31m",    # 红色
    "CRITICAL": "\033[35m", # 紫色
    "BOLD": "\033[1m",      # 粗体
}

# 级别映射
LEVEL_NAMES = {
    LOG_LEVEL_DEBUG: "DEBUG",
    LOG_LEVEL_INFO: "INFO",
    LOG_LEVEL_WARNING: "WARNING",
    LOG_LEVEL_ERROR: "ERROR",
    LOG_LEVEL_CRITICAL: "CRITICAL",
}

def ensure_log_dir_exists():
    """确保日志目录存在"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def get_log_filename():
    """获取基于当前时间的日志文件名"""
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"{timestamp}.txt")

# 全局变量，用于存储当前日志文件路径
current_log_file = None

def initialize_logger():
    """初始化日志记录器，创建新的日志文件"""
    global current_log_file
    ensure_log_dir_exists()
    current_log_file = get_log_filename()
    
    # 创建日志文件并写入初始信息
    with open(current_log_file, "w", encoding="utf-8") as f:
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"=== 日志开始于 {start_time} ===\n")
    
    return current_log_file

def set_log_level(level):
    """设置日志级别
    
    参数:
        level: 日志级别，可以是整数或字符串 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global CURRENT_LOG_LEVEL
    
    if isinstance(level, str):
        level = level.upper()
        if level == "DEBUG":
            CURRENT_LOG_LEVEL = LOG_LEVEL_DEBUG
        elif level == "INFO":
            CURRENT_LOG_LEVEL = LOG_LEVEL_INFO
        elif level == "WARNING":
            CURRENT_LOG_LEVEL = LOG_LEVEL_WARNING
        elif level == "ERROR":
            CURRENT_LOG_LEVEL = LOG_LEVEL_ERROR
        elif level == "CRITICAL":
            CURRENT_LOG_LEVEL = LOG_LEVEL_CRITICAL
        else:
            raise ValueError(f"未知的日志级别: {level}")
    else:
        CURRENT_LOG_LEVEL = level
    
    log_message(f"日志级别设置为: {LEVEL_NAMES.get(CURRENT_LOG_LEVEL, str(CURRENT_LOG_LEVEL))}", LOG_LEVEL_INFO)

def should_log(level):
    """判断是否应该记录该级别的日志"""
    return level >= CURRENT_LOG_LEVEL

def log_message(message, level=LOG_LEVEL_INFO):
    """将消息写入日志文件
    
    参数:
        message: 日志消息
        level: 日志级别 (默认为INFO)
    """
    global current_log_file
    
    # 检查是否应该记录该级别的日志
    if not should_log(level):
        return
    
    # 如果日志文件未初始化，则初始化
    if current_log_file is None:
        initialize_logger()
    
    # 获取当前时间
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 获取级别名称
    level_name = LEVEL_NAMES.get(level, str(level))
    
    # 将消息写入日志文件 (日志文件中不包含颜色代码)
    with open(current_log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level_name}] {message}\n")
    
    # 如果是终端输出，添加颜色
    if sys.stdout.isatty():
        color = COLORS.get(level_name, COLORS["RESET"])
        print(f"{color}[{timestamp}] [{level_name}] {message}{COLORS['RESET']}")

def debug(message):
    """记录DEBUG级别日志"""
    log_message(message, LOG_LEVEL_DEBUG)

def info(message):
    """记录INFO级别日志"""
    log_message(message, LOG_LEVEL_INFO)

def warning(message):
    """记录WARNING级别日志"""
    log_message(message, LOG_LEVEL_WARNING)

def error(message):
    """记录ERROR级别日志"""
    log_message(message, LOG_LEVEL_ERROR)

def critical(message):
    """记录CRITICAL级别日志"""
    log_message(message, LOG_LEVEL_CRITICAL)

# 为了向后兼容，保留原始log_message函数行为
def log_message_compat(message, level=None):
    """原始log_message函数的兼容实现，支持可选的级别参数"""
    # 如果没有提供级别，使用INFO级别
    if level is None:
        level = LOG_LEVEL_INFO
    # 调用新的log_message函数
    log_message_original(message, level)

# 保存原始函数的引用
log_message_original = log_message

# 覆盖原始的log_message函数以保持完全向后兼容
log_message = log_message_compat 