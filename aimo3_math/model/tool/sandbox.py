import contextlib
import queue
import os
import re
import threading
import time
from typing import Optional, List
from .base import ToolBase

from jupyter_client import KernelManager

tool_prompt = (
    'Use this tool to execute Python code for:\n'
    '- Complex calculations that would be error-prone by hand\n'
    '- Numerical verification of analytical results\n'
    '- Generating examples or testing conjectures\n'
    '- Visualizing problem structure when helpful\n'
    '- Brute-force verification for small cases\n\n'

    'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
    'Always use print() to display results. Write clear, well-commented code.\n\n'

    'Remember: Code should support your mathematical reasoning, not replace it. '
    'Explain what you\'re computing and why before running code.'
)


class AIMO3Sandbox(ToolBase):
    """
    AIMO3 沙箱环境：用于安全地执行 Python 代码片段。
    支持上下文管理器 (with 语句)，确保资源自动回收。
    """

    _port_lock = threading.Lock()
    _next_port = 50000
    name = "python_sandbox"

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> List[int]:
        """动态分配可用端口，避免多实例冲突"""
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):
        self._default_timeout = timeout
        self._km: Optional[KernelManager] = None
        self._client = None
        self._owns_kernel = False

        # 初始化内核
        self._setup_kernel()
        # 预加载基础库
        self.reset()

    def get_tool_schema(self):
        schema = {
            "type": "function",
            "function": {
                "name": "python_sandbox",
                "description": tool_prompt,  # 关键元信息：工具描述
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python script to execute",  # 关键元信息：参数描述
                        },
                        "timeout": {
                            "type": "float",
                            "description": "running timeout in seconds",
                        },
                    },
                    "required": ["code"],
                },
                "strict": True  # 开启严格模式，确保输出符合 Schema
            }
        }
        return schema


    def _setup_kernel(self):
        """初始化 Jupyter 内核配置"""
        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env.update({
            'PYDEVD_DISABLE_FILE_VALIDATION': '1',
            'PYDEVD_WARN_EVALUATION_TIMEOUT': '0',
            'JUPYTER_PLATFORM_DIRS': '1',
            'PYTHONWARNINGS': 'ignore',
            'MPLBACKEND': 'Agg'
        })

        self._km = KernelManager(
            shell_port=ports[0],
            iopub_port=ports[1],
            stdin_port=ports[2],
            hb_port=ports[3],
            control_port=ports[4],
            env=env
        )
        self._km.start_kernel()
        self._owns_kernel = True
        self._client = self._km.client()
        self._client.start_channels()

    def __enter__(self):
        """支持 with 语句开始"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句结束，自动关闭资源"""
        self.close()

    def __call__(self, code: str, timeout: Optional[float] = None):
        return self.execute(code, timeout)

    def execute(self, code: str, timeout: Optional[float] = None) -> str:
        """
        执行代码并捕获输出。
        """
        if not self._client:
            return "[ERROR] Sandbox is closed."

        effective_timeout = timeout or self._default_timeout
        self._client.execute(code)

        start_time = time.time()
        stdout_parts = []
        stderr_parts = []

        while True:
            # 检查是否超时
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                return f'[ERROR] Execution timed out after {effective_timeout} seconds.'

            try:
                # 轮询消息，设置较短的 timeout 以便能及时触发外部的总超时检查
                msg = self._client.get_iopub_msg(timeout=0.2)
                msg_type = msg['header']['msg_type']
                content = msg['content']

                if msg_type == 'stream':
                    target = stdout_parts if content['name'] == 'stdout' else stderr_parts
                    target.append(content['text'])

                elif msg_type == 'error':
                    stderr_parts.append(self._format_error(content['traceback']))

                elif msg_type in {'execute_result', 'display_data'}:
                    if 'data' in content and 'text/plain' in content['data']:
                        text = content['data']['text/plain']
                        stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

                elif msg_type == 'status' and content.get('execution_state') == 'idle':
                    break

            except queue.Empty:
                continue
            except Exception as e:
                stderr_parts.append(f"[INTERNAL ERROR] {str(e)}")
                break

        return self._build_output(stdout_parts, stderr_parts)

    def _build_output(self, stdout_list: List[str], stderr_list: List[str]) -> str:
        """整理并合并输出结果"""
        stdout = "".join(stdout_list).strip()
        stderr = "".join(stderr_list).strip()

        if stderr:
            return f"{stdout}\n[STDERR]\n{stderr}".strip()
        return stdout if stdout else "[WARN] No output. Use print() or expression at the end."

    def _format_error(self, traceback_list: List[str]) -> str:
        """格式化错误信息，去除干扰路径"""
        cleaned = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in traceback_list]
        # 过滤掉 IPython 内部逻辑的报错行
        filtered = [line for line in cleaned if 'IPython' not in line]
        return "\n".join(filtered)

    def close(self, suppress=contextlib.suppress):
        if self._client:
            with suppress(Exception):
                self._client.stop_channels()
                self._client = None

        if self._owns_kernel and self._km is not None:
            with suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with suppress(Exception):
                self._km.cleanup_resources()
            self._km = None
            self._owns_kernel = False

    def reset(self):
        """重置沙箱命名空间"""
        reset_code = (
            '%reset -f\n'
            'import math, numpy as np, sympy, itertools\n'
            'from mpmath import mp\n'
            'mp.dps = 64\n'
        )
        self.execute(reset_code)

    def __del__(self):
        """兜底清理"""
        self.close()