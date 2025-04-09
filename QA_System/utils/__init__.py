from utils.logger import (
    log_message, 
    debug, 
    info, 
    warning, 
    error, 
    critical, 
    set_log_level,
    initialize_logger,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL
)

# 为了保持向后兼容，继续导出log_message
__all__ = [
    'log_message',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'set_log_level',
    'initialize_logger',
    'LOG_LEVEL_DEBUG',
    'LOG_LEVEL_INFO',
    'LOG_LEVEL_WARNING',
    'LOG_LEVEL_ERROR',
    'LOG_LEVEL_CRITICAL'
] 