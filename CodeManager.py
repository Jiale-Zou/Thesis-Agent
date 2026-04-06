import re
import logging             # 日志记录（追踪沙箱行为、错误）
import difflib
from collections import Counter
from typing import Dict, List, Optional, Annotated
import subprocess
import tempfile            # 临时文件/目录（沙箱核心，这里没直接用但属于沙箱标准库）
import os
import shutil              # 文件操作（删除目录、清理沙箱）
import sys
import json
from pathlib import Path   # 现代路径操作（比os.path更安全、简洁）

class CodeSandbox:
    """
    安全的代码执行沙箱
    特性：
    1. 自动生成项目目录
    2. 自动安装requirements.txt
    3. 严格限制文件系统访问
    4. 防止沙箱逃逸
    """

    def __init__(
            self,
            base_workspace: str = "D:\\AiProgram\\Thesis",  # 沙箱根目录
            max_execution_time: int = 60,  # 代码最大执行时间：60秒
            max_output_size: int = 10000,  # 输出最大长度：10KB
            network_disabled: bool = True  # 默认禁用网络
    ):
        self.base_workspace = Path(base_workspace)  # 把路径转为Path对象（安全）
        self.base_workspace.mkdir(exist_ok=True, parents=True)  # 创建根目录，不存在则新建

        self.max_execution_time = max_execution_time
        self.max_output_size = max_output_size
        self.network_disabled = network_disabled

        # 安全黑名单（防止危险操作）—— 正则匹配危险代码
        self.dangerous_patterns = [
            r'os\.system',  # 禁止执行系统命令
            r'subprocess\.Popen.*shell=True',  # 禁止shell执行
            r'__import__.*os',  # 禁止动态导入os模块
            r'open\(.*\/etc\/',  # 禁止访问系统敏感目录
            r'rm\s+-rf',  # 禁止强制删除文件
            r'chmod\s+777',  # 禁止修改权限
            r'chown',  # 禁止修改所有者
            r'sudo',  # 禁止提权
            r'eval\(.*input',  # 禁止执行用户输入
            r'exec\(.*input',  # 禁止执行用户输入
        ]

        # 日志
        logging.basicConfig(level=logging.INFO)  # 设置日志级别为INFO，输出普通运行信息
        self.logger = logging.getLogger(__name__)  # 创建日志对象，用于打印沙箱运行日志

    def create_project(self, project_name: str = None) -> Path:
        """创建项目目录结构，返回项目根目录路径
        - 为每一次代码执行创建独立隔离目录
        - 目录结构标准化，防止文件混乱
        - 随机项目名避免冲突，提升安全性
        """
        if project_name is None:
            project_name = f"project_{os.urandom(4).hex()}"  # 随机生成项目名

        project_dir = self.base_workspace / project_name  # 拼接项目完整路径
        project_dir.mkdir(exist_ok=True)  # 创建项目目录
        self.project_dir = project_dir

        # 创建标准项目结构（隔离不同用途文件）
        (project_dir / "src").mkdir(exist_ok=True)  # 源码目录
        (project_dir / "data").mkdir(exist_ok=True)  # 数据目录
        (project_dir / "logs").mkdir(exist_ok=True)  # 日志目录
        (project_dir / "img").mkdir(exist_ok=True)  # 图片目录

        # 创建README说明文件
        readme_content = f"""# {project_name}

Auto-generated project directory for AI Agent code execution.

## Structure
- src/: Source code
- data/: Input/output data
- logs/: Execution logs
- img/: Image files
"""
        (project_dir / "README.md").write_text(readme_content)

        self.logger.info(f"Created project directory: {project_dir}")
        return project_dir  # 返回项目路径

    def install_dependencies(self, project_dir: Path, requirements_content: str) -> Dict:
        '''安装requirements.txt中的依赖
        - 安全安装 Python 库
        - 严格限制在沙箱目录内
        - 超时保护、异常捕获
        - 输出截断防止恶意输出
        '''
        if not requirements_content.strip():
            return {"success": True, "message": "No dependencies to install"}

        req_file = project_dir / "requirements.txt"
        req_file.write_text(requirements_content)  # 写入依赖文件

        self.logger.info(f"Installing dependencies from {req_file}")

        try:
            # 调用pip安装依赖
            # 1. subprocess.run返回：
            #   - result.returncode  # 退出码：0=成功，非0=失败
            #   - result.stdout      # 正常输出
            #   - result.stderr      # 错误输出
            # 2. sys.executable = 当前正在运行的 Python 解释器的完整绝对路径
            #   - 确保子进程使用和主程序完全一样的 Python 解释器
            # 3. subprocess传入env，就是给子进程造了一个独立、干净、受控的假系统环境
            #   -
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                capture_output=True,  # 捕获输出
                text=True,  # 文本模式
                timeout=300,  # 安装超时5分钟
                cwd=str(project_dir),  # 工作目录限制在项目内
                env=self._get_safe_env(project_dir)  # 使用安全环境变量
            )

            # 返回安装结果，截断超长输出
            if result.returncode == 0:
                self.logger.info("Dependencies installed successfully")
                return {
                    "success": True,
                    "stdout": result.stdout[:self.max_output_size],
                    "stderr": result.stderr[:self.max_output_size],
                    "exit_code": 0,
                }
            else:
                self.logger.error(f"Failed to install dependencies: {result.stderr}")
                return {
                    "success": False,
                    "stdout": result.stdout[:self.max_output_size],
                    "stderr": result.stderr[:self.max_output_size],
                    "exit_code": result.returncode,
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Dependency installation timeout (5 minutes)",
                "exit_code": 2,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Installation error: {str(e)}",
                "exit_code": -1,
            }

    def _get_safe_env(self, project_dir: Path) -> Dict:
        '''安全环境变量
        - 限制 Python 只能导入沙箱内代码，无法访问系统模块
        - 清空网络代理 = 强制断网，防止代码上传数据、下载木马
        '''
        safe_env = os.environ.copy()  # 复制系统环境变量
        safe_env['PYTHONPATH'] = str(project_dir / "src")  # 只允许导入沙箱内的模块

        if self.network_disabled:
            safe_env['http_proxy'] = ''
            safe_env['https_proxy'] = ''  # 清空代理 → 断网

        return safe_env

    def _validate_code_safety(self, code: str) -> Dict:
        '''代码安全扫描
        - 扫描代码是否包含黑名单操作
        - 禁止访问系统路径
        - 禁止 ../ 跳转上级目录（沙箱逃逸最常用方法）
        - 发现风险直接拒绝执行代码
        '''
        violations = []
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code):
                violations.append(f"Detected dangerous pattern: {pattern}")

        # 禁止访问系统绝对路径
        if re.search(r'\/(etc|root|home|var|usr|bin|sbin)', code):
            violations.append("Absolute system paths detected")

        # 禁止跳转到父目录（防止逃逸）
        if re.search(r'\.\.\/', code):
            violations.append("Parent directory access detected")

        if violations:
            return {
                "safe": False,
                "violations": violations
            }
        return {"safe": True, "violations": []}

    def _sanitize_path(self, path: Path, project_dir: Path) -> Path:
        '''强制所有文件操作必须在沙箱目录内，任何跳出沙箱的行为直接报错。'''

        abs_path = path.resolve() # 把传入的 path 解析成【绝对路径】
        project_abs = project_dir.resolve() # 把沙箱根目录也解析成【绝对路径】

        try:
            abs_path.relative_to(project_abs)  # 检查：传入的路径是否属于沙箱目录的子路径
            return abs_path
        except ValueError:
            raise ValueError(f"Path {path} is outside project directory {project_dir}")

    def execute_code(self, project_dir: Path, params: List, code: str, filename="main.py") -> Dict:
        # 1. 安全检查
        safety_check = self._validate_code_safety(code)
        if not safety_check["safe"]:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Security violation detected:\n" + "\n".join(safety_check["violations"]),
                "exit_code": -1,
                "security_violation": True
            }

        # 2. 校验路径是否合法
        try:
            project_dir = self._sanitize_path(project_dir, self.base_workspace)
        except ValueError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1
            }

        # 3. 把代码写入沙箱内的文件
        def clean_code(code: str) -> str:
            """清洗代码：
            1. 去掉多余缩进
            2. 去掉 ```python ``` 标记
            3. 去掉行首空格
            """
            import textwrap
            # 1. 去掉 markdown 代码块
            code = code.strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            # 2. 自动去掉 统一缩进
            code = textwrap.dedent(code)
            return code
        code = clean_code(code)
        code_file = project_dir / "src" / filename
        code_file.write_text(code, encoding='utf-8')

        self.logger.info(f"Executing code in {project_dir}")

        # 4. 执行代码
        try:
            process = subprocess.run(
                [sys.executable, str(code_file)] + params,  # 用当前Python运行代码
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,  # 超时杀死进程
                cwd=str(project_dir / "src"),  # 工作目录锁定src
                env=self._get_safe_env(project_dir),  # 安全环境（断网、限模块）
                encoding='utf-8', # window的subprocess输出默认使用gbk编码解析
            )

            # 5. 限制输出长度，返回结果
            stdout = process.stdout[:self.max_output_size]
            stderr = process.stderr[:self.max_output_size]

            # 检查是否有安全违规输出
            if "Permission denied" in stderr or "Operation not permitted" in stderr:
                return {
                    "success": False,
                    "stdout": stdout,
                    "stderr": "Security restriction: File system access denied",
                    "exit_code": process.returncode,
                    "security_violation": True
                }

            return {
                "success": process.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": process.returncode,
                "security_violation": False
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timeout ({self.max_execution_time}s limit)",
                "exit_code": -1,
                "timeout": True
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "stdout": e.stdout[:self.max_output_size] if e.stdout else "",
                "stderr": e.stderr[:self.max_output_size] if e.stderr else str(e),
                "exit_code": e.returncode,
                "security_violation": False
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "exit_code": -1,
                "security_violation": False
            }

    def cleanup_project(self, project_dir: Path):
        """清理项目目录"""
        try:
            if project_dir.exists() and project_dir.is_dir():
                project_dir = self._sanitize_path(project_dir, self.base_workspace) # 确保project_dir在base_workspace内
                shutil.rmtree(project_dir) # 删除整个project_dir
                self.logger.info(f"Cleaned up project: {project_dir}")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def get_project_structure(self, project_dir: Path) -> Dict:
        """获取项目结构信息
        - 遍历整个沙箱项目目录，把所有文件夹、所有文件、文件内容（安全截取）全部整理成一个字典返回
        """
        structure = {
            "project_dir": str(project_dir), # 项目根目录路径
            "files": [], # 所有文件列表（存这里）
            "directories": [] # 所有文件夹列表（存这里）
        }

        # os.walk(): 自动一层一层深入所有子目录
        #   - root：当前正在遍历的文件夹路径
        #   - dirs：当前文件夹里的所有子文件夹（列表）
        #   - files：当前文件夹里的所有文件（列表）
        for root, dirs, files in os.walk(project_dir):
            rel_root = Path(root).relative_to(project_dir) # 把绝对路径 → 转换成项目内的相对路径。

            for d in dirs:
                structure["directories"].append(str(rel_root / d))

            for f in files:
                file_path = Path(root) / f
                # 限制文件大小读取100KB
                if file_path.stat().st_size < 100000:  # file_path.stat().st_size: 获取文件大小（字节）
                    try:
                        structure["files"].append({
                            "path": str(rel_root / f),
                            "content": file_path.read_text()[:5000]  # 限制5000字符
                        })
                    except:
                        structure["files"].append({
                            "path": str(rel_root / f),
                            "content": "[Binary or unreadable file]"
                        })

        return structure


class CodeDiffManager:
    """代码差异管理器"""

    def get_diff_summary(self, old_code: str, new_code: str) -> Dict:
        """生成代码差异摘要"""

        # 使用unified_diff生成差异: Python 官方标准的代码对比工具
        #  - 逐行对比两个文本
        #  - (+ 开头 = 新增行)
        #  - (- 开头 = 删除行)
        diff = list(difflib.unified_diff(
            old_code.splitlines(keepends=True), # 文本序列需要按行分割，通常使用splitlines(keepends=True)保留换行符
            new_code.splitlines(keepends=True),
            fromfile='old_version', # 旧版本文件名（标记用）
            tofile='new_version', # 新版本文件名（标记用）
            lineterm=''  # 不额外加换行符
        ))

        # 提取关键信息
        added_lines = [] # 新增的代码行
        removed_lines = [] # 删除的代码行
        changed_functions = set() # 被修改的函数名

        for line in diff:
            # 以 + 开头，且不是文件头 +++
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:].strip()) # 去掉 + 号，存入新增行
                # 如果这一行是 def 开头 → 代表函数被修改
                if line.strip().startswith('def '):
                    func_name = line.split('def ')[1].split('(')[0]
                    changed_functions.add(func_name)
            # 以 - 开头，且不是文件头 ---
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines.append(line[1:].strip())
                # 检测被删除的函数
                if line.strip().startswith('def '):
                    func_name = line.split('def ')[1].split('(')[0]
                    changed_functions.add(func_name)

        return {
            'has_changes': len(diff) > 0, # 是否有变化
            'diff_lines': len(diff)/2, # 差异总行数
            'added_lines_count': len(added_lines), # 新增行数
            'removed_lines_count': len(removed_lines), # 删除行数
            'changed_functions': list(changed_functions), # 修改的函数
            'diff_preview': '\n'.join(diff[:50])  # 限制预览长度
        }


class ErrorSummarizer:
    """错误信息摘要器"""

    def __init__(self):
        # Python常见错误模式
        self.error_patterns = {
            'syntax_error': r'SyntaxError: (.+)',
            'name_error': r"NameError: name '(.+)' is not defined",
            'type_error': r'TypeError: (.+)',
            'attribute_error': r"AttributeError: '(.+)' object has no attribute '(.+)'",
            'index_error': r'IndexError: (.+)',
            'key_error': r"KeyError: (.+)",
            'import_error': r'(ImportError|ModuleNotFoundError): (.+)',
            'value_error': r'ValueError: (.+)',
        }
        # pip常见错误模式
        self.pip_patterns = {
            'pip_not_found': r'pip: command not found|pip is not recognized',
            'package_not_found': r'Could not find a version that satisfies the requirement|No matching distribution found',
            'network_error': r'Could not fetch URL|Connection refused|Timeout|connect failed|network is unreachable',
            'permission_denied': r'PermissionError|permission denied|Access denied',
            'version_conflict': r'conflict|VersionConflict|requires a different version',
            'invalid_requirement': r'Invalid requirement|error in setup command',
            'out_of_memory': r'MemoryError|Killed|std::bad_alloc',
        }

    def extract_error_type(self, stderr: str) -> Dict:
        """提取错误类型和关键信息"""
        error_info = {
            'error_type': 'Unknown', # 错误类型（语法/变量/类型等）
            'error_source': 'unknown',  # code / pip
            'error_message': '', # 错误详情
            'line_number': None, # 出错行号
            'affected_component': '' # 出问题的组件（变量名/属性名）
        }

        ## 检测错误类型 ##
        # ======================
        # 先判断是不是 pip 错误
        # ======================
        for err_type, pattern in self.pip_patterns.items():
            if re.search(pattern, stderr, re.IGNORECASE):
                error_info['error_type'] = err_type
                error_info['error_source'] = 'pip'
                error_info['error_message'] = self._clean_pip_error(stderr)
                return error_info
        # ======================
        # 再判断是不是代码错误
        # =====================
        for error_type, pattern in self.error_patterns.items():
            match = re.search(pattern, stderr, re.IGNORECASE)
            if match:
                error_info['error_type'] = error_type
                error_info['error_message'] = match.group(1)[:200]  # 限制长度
                # 提取行号
                line_match = re.search(r'line (\d+)', stderr)
                if line_match:
                    error_info['line_number'] = int(line_match.group(1))
                # 提取受影响的组件
                if error_type == 'name_error' and len(match.groups()) >= 1:
                    error_info['affected_component'] = match.group(1)
                elif error_type == 'attribute_error' and len(match.groups()) >= 2:
                    error_info['affected_component'] = f"{match.group(1)}.{match.group(2)}"
                elif error_type == 'import_error':
                    error_info['affected_component'] = match.group(2)[:30]
                break

        # 都没匹配到
        error_info['error_message'] = stderr[:200]

        return error_info

    def _clean_pip_error(self, stderr: str) -> str:
        """清理 pip 错误，只保留关键信息(指出未pip模块的行)"""
        lines = stderr.splitlines()
        for line in lines:
            if any(key in line for key in ['ERROR', 'Warning', 'Failed', 'Could not']):
                return line.strip()[:150]
        return stderr[:150]


    def summarize_errors(self, error_list: List[str], max_errors: int = 5) -> str:
        """汇总多个错误"""

        if not error_list:
            return "No errors"

        # 只取最近的几个错误
        recent_errors = error_list[-max_errors:]

        summaries = []
        error_types = Counter()

        for error in recent_errors:
            info = self.extract_error_type(error)
            error_types[info['error_type']] += 1 # 统计错误类型次数

            # 生成简洁描述
            src = f"[{info['error_source']}]" if info['error_source'] != 'unknown' else "" # 错误源: pip / code
            msg = info['error_message'] # 详细错误信息
            line = f" (line {info['line_number']})" if info['line_number'] else "" # 错误行号
            summaries.append(f"{src}{info['error_type']}: {msg}{line}")


        # 统计错误类型分布
        type_stats = ", ".join([f"{k}({v})" for k, v in error_types.items()])

        return f"Errors [{type_stats}]: " + " | ".join(summaries)[:500]