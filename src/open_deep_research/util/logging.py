"""
Structured Logging System

Provides unified logging functionality:
- Structured log format
- Multiple output targets (console, file, remote)
- Log level control
- Performance monitoring integration
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import os

import structlog
from structlog.stdlib import LoggerFactory

from .config import settings
from .config import Environment

# Global flag to prevent duplicate configuration
_logging_configured = False
_file_logging_configured = False


def configure_logging(force_file_logging: bool = False) -> None:
    """Configure the structured logging system.

    Args:
        force_file_logging: Force file logging, ignoring debug mode.
    """
    global _logging_configured
    
    # Prevent duplicate configuration
    if _logging_configured:
        return
        
    # Detect if running in stdio MCP server mode
    is_stdio_mode = force_file_logging or os.getenv("COREMCP_STDIO_MODE", "").lower() in ("true", "1", "yes")
    
    # In stdio mode, completely disable console output and only use file logging
    if is_stdio_mode:
        # Configure a processor with no console output for stdio mode
        class EventOnlyRenderer:
            """A simplified renderer that only renders the event content."""
            def __call__(self, logger, method_name, event_dict):
                # Only keep the event content, remove other structured information
                if 'event' in event_dict:
                    return event_dict['event']
                # If there is no event field, return the values of all non-metadata fields
                filtered_dict = {k: v for k, v in event_dict.items() 
                                #  timestamp='2025-06-26T08:25:56.452809Z' level='info' logger='xxx_logging' event='xxx' filename='xxx.py' func_name='xxx' lineno=34
                               if k not in ['timestamp', 'level', 'logger', 'filename', 'func_name', 'lineno']}
                if filtered_dict:
                    return ' '.join(str(v) for v in filtered_dict.values())
                return str(event_dict)
        
        processors = [
            # Add timestamp
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            # Add call site information
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            # Format exceptions
            structlog.processors.format_exc_info,
            # Use a custom event renderer to display only useful information
            EventOnlyRenderer(),
        ]
    else:
        # Processor configuration for normal mode
        processors = [
            # Add timestamp
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            # Add call site information
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            # Format exceptions
            structlog.processors.format_exc_info,
            # JSON formatting (production) or colorized output (development)
            structlog.dev.ConsoleRenderer(colors=settings.debug)
            if settings.debug
            else structlog.processors.JSONRenderer(),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    if is_stdio_mode:
        # stdio mode: completely disable console output to avoid polluting stdio
        logging.basicConfig(
            level=getattr(logging, settings.log_level.value),
            handlers=[]  # Do not add any default handlers
        )
        # Remove all existing handlers from the root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    else:
        # Output to stderr in normal mode
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stderr,
            level=getattr(logging, settings.log_level.value),
        )
    
    # Configure file logging
    if not settings.debug or is_stdio_mode:
        setup_file_logging()
    
    # Mark as configured
    _logging_configured = True


def setup_file_logging() -> None:
    """Set up file logging."""
    global _file_logging_configured
    
    # Prevent duplicate file log configuration
    if _file_logging_configured:
        return
    
    # Ensure logs directory exists
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = settings.logs_dir / "core.log"
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Check if a file handler with the same name already exists
    existing_file_handlers = [
        h for h in root_logger.handlers 
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file.absolute())
    ]
    
    if existing_file_handlers:
        # The same file handler already exists, no need to add it again
        _file_logging_configured = True
        return
    
    # Create file handler, ensuring UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, settings.log_level.value))
    
    # Create a custom formatter to handle structlog formatted messages
    class StructlogCompatibleFormatter(logging.Formatter):
        """A structlog-compatible custom formatter to avoid duplicate information."""
        
        def __init__(self):
            super().__init__(
                # '%(asctime)s - %(levelname)s - %(name)s - %(pathname)s - %(lineno)d - %(message)s',
                '%(asctime)s - %(levelname)s - [Process - %(process)d - %(taskName)s] - [Thread - %(thread)d - %(threadName)s] - %(pathname)s - %(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        def format(self, record):
            # If the message is a structlog formatted key-value pair string, extract the event content
            msg = record.getMessage()
            
            # Check if it is a structlog formatted message
            if "event=" in msg and "timestamp=" in msg:
                # Extract event content
                import re
                event_match = re.search(r"event='([^']*)'", msg)
                if event_match:
                    # Only keep event content, remove duplicate structured information
                    record.msg = event_match.group(1)
                    record.args = ()
            
            # Format using the standard formatter
            formatted = super().format(record)
            
            # Ensure Chinese characters are displayed correctly, remove possible control characters
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            formatted = ansi_escape.sub('', formatted)
            
            return formatted
    
    formatter = StructlogCompatibleFormatter()
    file_handler.setFormatter(formatter)
    
    # Create a filter to ensure correct message encoding and remove ANSI control characters
    class UnicodeANSIFilter(logging.Filter):
        def filter(self, record):
            # Ensure the message is a correct Unicode string
            if hasattr(record, 'msg') and record.msg is not None:
                # If it's a byte string, try to decode it
                if isinstance(record.msg, bytes):
                    try:
                        record.msg = record.msg.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            record.msg = record.msg.decode('gbk')
                        except UnicodeDecodeError:
                            record.msg = record.msg.decode('utf-8', errors='replace')
                
                # Ensure it is a string type
                if not isinstance(record.msg, str):
                    record.msg = str(record.msg)
                
                # Remove ANSI color control characters
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                record.msg = ansi_escape.sub('', record.msg)
            
            return True
    
    file_handler.addFilter(UnicodeANSIFilter())
    
    # Add to the root logger
    root_logger.addHandler(file_handler)
    
    # Mark file logging as configured
    _file_logging_configured = True


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger.
    
    Args:
        name: Logger name, defaults to the calling module name.
        
    Returns:
        A configured structured logger.
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Logger mixin class to provide logging functionality to other classes."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get the logger for the current class."""
        return get_logger(self.__class__.__name__)


class RequestLogger:
    """Request logger."""
    
    def __init__(self):
        self.logger = get_logger("request")
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log an HTTP request."""
        self.logger.info(
            "HTTP Request",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            user_id=user_id,
            **kwargs
        )
    
    def log_error(
        self,
        method: str,
        path: str,
        error: Exception,
        user_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log a request error."""
        self.logger.error(
            "HTTP Request Error",
            method=method,
            path=path,
            error=str(error),
            error_type=type(error).__name__,
            user_id=user_id,
            **kwargs,
            exc_info=True
        )


class PerformanceLogger:
    """Performance monitoring logger."""
    
    def __init__(self):
        self.logger = get_logger("performance")
    
    def log_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        **kwargs: Any
    ) -> None:
        """Log operation performance."""
        self.logger.info(
            "Operation Performance",
            operation=operation,
            duration_ms=round(duration * 1000, 2),
            success=success,
            **kwargs
        )
    
    def log_database_query(
        self,
        query: str,
        duration: float,
        rows_affected: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Log database query performance."""
        self.logger.info(
            "Database Query",
            query=query[:100] + "..." if len(query) > 100 else query,
            duration_ms=round(duration * 1000, 2),
            rows_affected=rows_affected,
            **kwargs
        )


class SecurityLogger:
    """Security event logger."""
    
    def __init__(self):
        self.logger = get_logger("security")
    
    def log_authentication(
        self,
        user_id: str,
        success: bool,
        method: str = "password",
        ip_address: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log authentication events."""
        self.logger.info(
            "User Authentication",
            user_id=user_id,
            success=success,
            method=method,
            ip_address=ip_address,
            **kwargs
        )
    
    def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        **kwargs: Any
    ) -> None:
        """Log authorization events."""
        self.logger.info(
            "Permission Check",
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            **kwargs
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log security events."""
        log_method = getattr(self.logger, severity.lower(), self.logger.warning)
        log_method(
            "Security Event",
            event_type=event_type,
            severity=severity,
            description=description,
            user_id=user_id,
            ip_address=ip_address,
            **kwargs
        )


# Global logger instances
request_logger = RequestLogger()
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()

# Initialize logging system
# Check environment variables or force file logging in production environment
force_file_logging = (
    os.getenv("COREMCP_FORCE_FILE_LOGGING", "").lower() in ("true", "1", "yes") or
    settings.environment != Environment.DEVELOPMENT
)
configure_logging(force_file_logging=force_file_logging)


def reset_logging_config() -> None:
    """Reset logging configuration state, mainly for testing environments."""
    global _logging_configured, _file_logging_configured
    _logging_configured = False
    _file_logging_configured = False
    
    # Clear all log handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close() 