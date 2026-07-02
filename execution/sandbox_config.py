"""
Configuration constants for sandboxed code execution.

Centralises Docker image names, resource limits, timeouts, and
per-language compilation/execution settings used by
:mod:`execution.sandbox`.
"""

import os

# =============================================================================
# Docker image
# =============================================================================
DOCKER_IMAGE_NAME = 'hce-sandbox:latest'

# =============================================================================
# Resource limits
# =============================================================================
DEFAULT_MEMORY_LIMIT = '256m'
JAVA_MEMORY_LIMIT = '512m'  # JVM needs more headroom
DEFAULT_CPU_LIMIT = 0.5

# =============================================================================
# Timeouts (seconds)
# =============================================================================
DEFAULT_TIMEOUT = 10
COMPILED_LANG_TIMEOUT = 20  # Java/C++ need compile + run time

# =============================================================================
# Output limits
# =============================================================================
MAX_OUTPUT_LENGTH = 10_000

# =============================================================================
# Sandbox user
# =============================================================================
SANDBOX_USER = 'sandbox'

# =============================================================================
# tmpfs size for sandbox work directory
# =============================================================================
TMPFS_SIZE = '64m'

# =============================================================================
# Docker requirement flag
# =============================================================================
# When True, refuse to execute if Docker is not available.
REQUIRE_DOCKER = os.environ.get('REQUIRE_DOCKER', '').lower() in ('true', '1', 'yes')

# =============================================================================
# Supported languages and their configurations
# =============================================================================
LANGUAGE_CONFIG = {
    'python': {
        'extension': '.py',
        'filename': 'user_script.py',
        'compile_cmd': None,
        'run_cmd': ['python3', '/tmp/sandbox/user_script.py'],
        'memory_limit': DEFAULT_MEMORY_LIMIT,
        'timeout': DEFAULT_TIMEOUT,
    },
    'javascript': {
        'extension': '.js',
        'filename': 'user_script.js',
        'compile_cmd': None,
        'run_cmd': ['node', '/tmp/sandbox/user_script.js'],
        'memory_limit': DEFAULT_MEMORY_LIMIT,
        'timeout': DEFAULT_TIMEOUT,
    },
    'java': {
        'extension': '.java',
        'filename': 'Main.java',
        'compile_cmd': ['javac', '/tmp/sandbox/Main.java'],
        'run_cmd': ['java', '-Xmx384m', '-cp', '/tmp/sandbox', 'Main'],
        'memory_limit': JAVA_MEMORY_LIMIT,
        'timeout': COMPILED_LANG_TIMEOUT,
    },
    'cpp': {
        'extension': '.cpp',
        'filename': 'main.cpp',
        'compile_cmd': ['g++', '-o', '/tmp/sandbox/main', '/tmp/sandbox/main.cpp'],
        'run_cmd': ['/tmp/sandbox/main'],
        'memory_limit': DEFAULT_MEMORY_LIMIT,
        'timeout': COMPILED_LANG_TIMEOUT,
    },
}
