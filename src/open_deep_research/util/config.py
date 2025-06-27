"""
Configuration Management Module

Provides a layered configuration system that supports:
- Default configuration
- Environment variables
- Configuration files
- Dynamic configuration updates
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Environment type enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentMode(str, Enum):
    """Deployment mode enumeration."""
    STANDALONE = "standalone"  # Standalone mode
    TEAM = "team"             # Team mode
    HYBRID = "hybrid"         # Hybrid mode


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application configuration class."""
    
    # ==================== Basic Configuration ====================
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    # log_level: LogLevel = LogLevel.INFO
    log_level: LogLevel = LogLevel.DEBUG
    
    # Project Information
    project_name: str = "OPEN_DEEP_RESEARCH"
    project_description: str = "Open Deep Research is an experimental, fully open-source research assistant that automates deep research and produces comprehensive reports on any topic. It's designed to help researchers, analysts, and curious individuals generate detailed, well-sourced reports without manual research overhead."
    version: str = "0.1.0"
    
    # ==================== Service Port Configuration ====================
    api_port: int = 8000
    web_ui_port: int = 3000
    prometheus_port: int = 9090
    grafana_port: int = 3001
    metrics_port: int = 8001
    
    # ==================== Database Configuration ====================
    # SQLite (Standalone mode)
    sqlite_database_path: str = "./data/core.db"
    
    # PostgreSQL (Team mode)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "core"
    postgres_user: str = "core"
    postgres_password: str = "core_dev_password"
    
    @property
    def database_url(self) -> str:
        """Returns the database URL based on the deployment mode."""
        if self.deployment_mode == DeploymentMode.STANDALONE:
            return f"sqlite:///{self.sqlite_database_path}"
        else:
            return (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
    
    # ==================== Cache Configuration ====================
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # ==================== Message Queue Configuration ====================
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "core"
    rabbitmq_password: str = "core_dev_password"
    rabbitmq_vhost: str = "/"
    
    @property
    def rabbitmq_url(self) -> str:
        """RabbitMQ connection URL."""
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}{self.rabbitmq_vhost}"
        )
    
    # ==================== Security Configuration ====================
    jwt_secret_key: str = "PRO1234567890"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    encryption_key: str = "xmjWxHyQd0ZZ8i6eoqUFs_DOkVKJTDy8SKLISk2AvrY="
    
    @validator("encryption_key")
    def validate_encryption_key(cls, v):
        # Fernet key must be a 44-character base64 encoded string
        if len(v) != 44 or not v.endswith('='):
            raise ValueError("The encryption key must be a valid Fernet base64 encoded key.")
        return v
    
    # ==================== Service Discovery Configuration ====================
    service_discovery_enabled: bool = True
    service_discovery_port: int = 5353
    service_name: str = "Core"
    service_type: str = "_core._tcp.local."
    
    # ==================== AI Model Configuration ====================
    # OpenAI API
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"
    
    # Local Model Configuration
    local_model_enabled: bool = False
    local_model_path: str = "./models/"
    local_model_api_url: str = "http://localhost:11434"
    
    # ==================== File Storage Configuration ====================
    upload_dir: str = "./uploads"
    max_file_size: int = 10485760  # 10MB
    allowed_file_types: List[str] = [
        ".txt", ".md", ".json", ".yaml", ".yml", 
        ".py", ".js", ".ts", ".vue"
    ]
    
    # ==================== Monitoring Configuration ====================
    prometheus_enabled: bool = True
    health_check_interval: int = 30
    
    # Sentry (Error Monitoring)
    sentry_dsn: Optional[str] = None
    sentry_environment: Optional[str] = None
    
    # ==================== Development Configuration ====================
    hot_reload: bool = True
    watch_files: bool = True
    profiling_enabled: bool = False
    sql_echo: bool = False
    
    # ==================== Deployment Mode ====================
    deployment_mode: DeploymentMode = DeploymentMode.STANDALONE
    
    # Cluster Configuration (Team Mode)
    cluster_enabled: bool = False
    cluster_nodes: List[str] = ["localhost:8000"]
    cluster_secret: str = "your_cluster_secret_here"
    
    # ==================== Third-Party Services ====================
    # Mail Service
    mail_enabled: bool = False
    mail_server: str = "smtp.gmail.com"
    mail_port: int = 587
    mail_username: Optional[str] = None
    mail_password: Optional[str] = None
    mail_from: Optional[str] = None
    mail_tls: bool = True
    mail_ssl: bool = False
    
    # Object Storage (Optional)
    s3_enabled: bool = False
    s3_bucket: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_endpoint: str = "https://s3.amazonaws.com"
    
    # ==================== Performance Optimization ====================
    worker_processes: int = 4
    worker_connections: int = 1000
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000
    db_pool_size: int = 10
    db_max_overflow: int = 20
    
    # ==================== Feature Flags ====================
    experimental_features: bool = False
    api_version: str = "v1"
    api_versioning_enabled: bool = True
    rate_limiting_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    # ==================== Path Configuration ====================
    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return Path(__file__).parent.parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Data directory."""
        data_dir = self.project_root / "data"
        data_dir.mkdir(exist_ok=True)
        return data_dir
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory."""
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        return logs_dir
    
    @property
    def uploads_dir(self) -> Path:
        """Upload files directory."""
        uploads_dir = Path(self.upload_dir)
        if not uploads_dir.is_absolute():
            uploads_dir = self.project_root / uploads_dir
        uploads_dir.mkdir(exist_ok=True)
        return uploads_dir
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields
        
        # Remove environment variable prefix to be compatible with existing API key settings
        # env_prefix = "CORE_"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings."""
    global settings
    settings = Settings()
    return settings 