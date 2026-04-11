import asyncio
import os
import re
import json
import platform
import string
from operator import add # 配合Annotated使用，将里面的元素用+合并值

import pandas as pd
from anthropic import BaseModel
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from pydantic import Field, field_validator
from sqlalchemy import TypeDecorator

from CodeManager import CodeDiffManager, ErrorSummarizer, CodeSandbox

from typing import TypedDict, Annotated, List, Optional, Dict, get_args
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


def excel_process(path):
    try:
        file = pd.read_excel(path, header=0)
    except:
        try:
            file = pd.read_excel(path, header=0, engine='xlrd')
        except Exception as e:
            try:
                file = pd.read_excel(path, header=0, engine='openpyxl')
            except:
                raise SystemError(f'ExcelError: {path} 文件损坏，打开失败。')
    ret = []
    columns = file.columns.tolist()
    dtypes = file.dtypes
    describe = file.describe(include=['object', 'bool', 'int', 'float'])
    for col in columns:
        ret.append(
            f'- {col}({dtypes[col]}) 描述性统计:{describe[col].to_dict()} | NaN数量:{file[col].isnull().sum()}{f" | 唯一元素:{file[col].unique()}" if dtypes[col] == "object" else ""}'
        )
    return '\n'.join(ret)
def csv_process(path):
    try:
        file = pd.read_csv(path, header=0, encoding='utf-8')
    except:
        try:
            file = pd.read_csv(path, header=0, encoding='gbk')
        except Exception as e:
            raise SystemError(f'CsvError: {path} 文件损坏，打开失败。')
    ret = []
    columns = file.columns.tolist()
    dtypes = file.dtypes
    describe = file.describe(include=['object', 'bool', 'int', 'float'])
    for col in columns:
        ret.append(
            f'- {col}({dtypes[col]}) 描述性统计:{describe[col].to_dict()} | NaN数量:{file[col].isnull().sum()}{f" | 唯一元素:{file[col].unique()}" if dtypes[col] == "object" else ""}'
        )
    return '\n'.join(ret)
def process(path):
    '''
    此工具用于查询指定csv或xlsx文件的元数据
    返回字符串，描述了字段名称、类型、描述性统计、NaN值的数量
    '''
    end = str(path).split('.')[1]
    if end == 'csv':
        ret = csv_process(path)
    elif end == 'xlsx':
        ret = excel_process(path)
    else:
        raise ValueError(f'传入的文件后缀为{end}，但只能处理csv和xlsx后缀的文件。')
    return ret


def clean_json_response(text: str) -> str:
    """
    清理模型响应，移除代码块标记
    """
    # 移除 ```json 和 ``` 标记
    cleaned = re.sub(r'```(?:json)?\n?|\n?```', '', text)

    # 移除可能的多余空白字符
    cleaned = cleaned.strip()

    # 如果开头是 {，确保结尾是 }（处理可能的截断）
    if cleaned.startswith('{') and not cleaned.endswith('}'):
        # 尝试找到最后一个 }
        last_brace = cleaned.rfind('}')
        if last_brace != -1:
            cleaned = cleaned[:last_brace + 1]

    return cleaned


class CodeAgentState(TypedDict):
    files: List[str] # 读取的文件路径
    step: int # 建模逻辑步骤数

    # 建模逻辑（精简描述）
    modeling_logic: str

    # 文件字段说明
    data_fields: str

    # 当前代码状态
    requirements: str
    current_code: str
    paths: List[str]

    # 上一轮代码状态
    prev_code: str

    # 执行结果
    execution_result: Annotated[
        List[Dict[str, str]],
        lambda left, right: (left or []) + (right or [])
    ]  # {stdout, stderr, exit_code}

    # 结果检查
    valid_reason: str

    # 当前运行状态
    state: str # debug, success, complete, continue

    # 迭代控制
    iteration_count: int
    max_iterations: int
class CodeGenerationAgent:
    def __init__(self, model, sandbox):
        # 初始化LLM
        self.llm = model
        # 初始化检查点（支持多轮对话）
        self.checkpointer = MemorySaver()
        # 构建状态图
        self.graph = self._build_graph()
        # 构建沙盒
        self.sandbox = sandbox
        self.project_dir = self.sandbox.project_dir
        # 初始化 代码管理器 和 报错管理器
        self.codeManager = CodeDiffManager()
        self.errorManager = ErrorSummarizer()

    def _build_graph(self):
        workflow = StateGraph(CodeAgentState)

        # 添加节点
        workflow.add_node("parse_node", self.parse_node)
        workflow.add_node("generate_code", self.generate_code)
        workflow.add_node("execute_code", self.execute_code)
        workflow.add_node("debug_and_fix", self.debug_and_fix)
        workflow.add_node("evaluate_result", self.evaluate_result)

        # 设置入口
        workflow.set_entry_point("parse_node")

        # 添加边
        workflow.add_edge("parse_node", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_conditional_edges(
            "execute_code",
            self.should_debug,
            {
                "success": "evaluate_result",
                "debug": "debug_and_fix",
                "max_iterations": END,
            }
        )
        workflow.add_conditional_edges(
            "evaluate_result",
            self.should_continue,
            {
                "continue": "execute_code",
                "complete": END,
            }
        )
        workflow.add_edge("debug_and_fix", "execute_code")
        # 编译图
        return workflow.compile(checkpointer=self.checkpointer)

    def should_debug(self, state: CodeAgentState) -> str:
        """决定是否需要调试"""
        if state.get("iteration_count", 0) >= state.get("max_iterations", 5):
            return "max_iterations"
        return state['state']

    def should_continue(self, state: CodeAgentState) -> str:
        """决定是否继续优化"""
        return state['state']

    def parse_node(self, state: CodeAgentState) -> CodeAgentState:
        print(f'  ===> code_step{state["step"]}: 开始解析文件。')

        ret = []
        files = state['files']
        for file in files:
            desc = process(file)
            ret.append(
                f'文件:{file}\n{desc}'
            )
        ret = '\n'.join(ret)
        print(ret)
        return {"data_fields": ret}

    def generate_code(self, state: CodeAgentState) -> CodeAgentState:
        """生成Python代码"""
        print(f'  ===> code_step{state["step"]}: 开始生成代码。')

        prompt = f"""
        你是一个专业的Python代码生成器。Python版本为{platform.python_version()}。
        
        *任务*
        你需要读取文件，文件的字段解释在*字段描述*中，你需要根据*建模逻辑*，写出可执行的python文件。
        
        *字段描述*
        {state['data_fields']}
        
        *建模逻辑*
        {state['modeling_logic']}
        
        *要求*
        1. 生成两个文件：依赖库文件requirements.txt和python可执行文件main.py。
        2. requirements.txt要求:
            - 需要更具python{platform.python_version()}版本选择不冲突的库。
            - 尽量选择简单且使用人数多的库，以免出现bug。
            - 必须是main中使用到的库。
        3. main.py要求:
            - 需要使用sys.argv依次接收两个参数: '结果csv文件输出目录'str,'图片输出目录'str
            - 若*建模逻辑*里需要生成最终的文件，请将该文件放在'结果csv文件输出目录'下，并同样以csv格式保存
            - 添加必要的注释，不需要设置错误处理机制，让bug充分暴露。
            - 只能使用*字段描述*中提及的字段名，不对数据的数量/质量、运行的效果进行检查。
            - 尽量选择简单且使用人数多的库，若遇到比较复杂或频繁bug的库，需考虑手动实现该功能。
            - 必须导入包“plt.rcParams['font.sans-serif'] = ['SimHei'] plt.rcParams['axes.unicode_minus'] = False”
            - 若需要保存dataframe，需确保保存的每一列都有名字(如果也要保存索引，则索引也得有名字)
            - 路径分隔符无要求，推荐使用os.path.join
            - 将执行过程的每一步按照指定格式print出来:
                ①使用编号"12..."标记步骤名称，如"1.数据读取"，"2.数据预览"
                ②说明该步骤的具体操作，如"读取了csv文件"，"使用describe进行了描述性统计"，"使用ARMA模型对字段year, gdp进行了建模"
                ③使用output标识该步骤的结果，如"output:共100个样本", "output:描述性统计的结果为...", "output:相关性热力图输出到了路径D:/data/img.jpg"
                ④每一步的完整格式为"1.相关性分析 n使用numpy对字段gdp和population进行了相关性分析，并将结果以热力图的方式可视化。output:热力图输出到了路径D:/data/img.jpg"
            - 输出:
                ①适当画图可视化，将图片输出到'图片输出目录'，图片以"步骤名称+图片含义"的方式命名，以jpg的格式输出。
                ②最终生成的结果文件名(csv文件)(只包含文件名，如xxx.csv)
        
        *JSON格式要求*
        1. 字符串必须用双引号包裹
        2. 字符串中的双引号必须转义
        3. 对象键必须用双引号包裹
        4. 每个属性后必须有逗号（除了最后一个）
        5. 布尔值必须是 true 或 false（小写）
        6. 中文字符可以直接使用，但特殊字符需要转义
        
        *返回格式*
        JSON是最外层且唯一的输出。
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:
        {{"requirements": "依赖库文件requirements", "python": "python可执行文件main", "path": ["若需要保存csv文件，保存的文件名称(csv格式)"]}}
        """
        prompts = [
            {'role': 'system', 'content': prompt},
        ]
        class Code(BaseModel):
            requirements: str = Field(description='依赖库文件requirements.txt', default="")
            python: str = Field(description='python可执行文件main.py', default="")
            path: List[str] = Field(description='若需要保存csv文件，保存的文件名称,储存在List里', default=[])

            @field_validator('requirements')
            @classmethod
            def requirements_check(cls, r: str) -> str:
                """开头不是字母，则报错"""
                if len(r) != 0:
                    first_char = r[0]
                    # 判断是否为空白字符（空格、制表符、换行等）
                    if first_char.isspace():
                        raise ValueError('格式错误: requirements文件以空白字符起始')
                    # 判断是否为标点符号
                    if first_char in string.punctuation:
                        raise ValueError('格式错误: requirements文件以标点符号起始')
                return r

            @field_validator('python')
            @classmethod
            def python_check(cls, p: str) -> str:
                """开头不是import from"""
                if len(p) != 0:
                    first_char = p[0]
                    # 判断是否为空白字符（空格、制表符、换行等）
                    if first_char.isspace():
                        raise ValueError('格式错误: python文件格式有误，以空白字符起始，预期以import/from起始')
                    # 判断是否为标点符号
                    if first_char in string.punctuation:
                        raise ValueError('格式错误: python文件格式有误，以标点符号起始，预期以import/from起始')
                return p

        raw_response = self.llm.invoke(prompts)
        raw_content = raw_response.content
        cleaned_content = clean_json_response(raw_content)  # 清理响应
        data = json.loads(cleaned_content)
        response = Code(**data)

        return {
            "requirements": response.requirements,
            "current_code": response.python,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "paths": response.path,
        }

    def execute_code(self, state: CodeAgentState) -> CodeAgentState:
        """在沙箱中执行代码"""
        print(f'  ===> code_step{state["step"]}: 开始执行代码。')
        install_res = self.sandbox.install_dependencies(self.project_dir, state['step'], state["requirements"])
        if install_res['success']:
            execute_res = self.sandbox.execute_code(self.project_dir, [(self.project_dir/"data"), (self.project_dir/"img")], state["current_code"], state['step'])
            return {
                "execution_result": [
                    {
                        "stdout": execute_res['stdout'],
                        "stderr": execute_res['stderr'],
                        "exit_code": execute_res['exit_code']
                    }
                ],
                "state": "success" if execute_res['success'] else "debug"
            }
        else:
            return {
                    "execution_result": [
                        {
                            "stdout": install_res['stdout'],
                            "stderr": install_res['stderr'],
                            "exit_code": install_res['exit_code']
                        }
                    ],
                    "state": "debug",
                }

    def debug_and_fix(self, state: CodeAgentState) -> CodeAgentState:
        """调试和修复"""

        print(f'  ===> code_step{state["step"]}: 开始调试和修复代码(Iteration {state.get("iteration_count", 0)})。')
        print(state['execution_result'][-1]['stderr'])

        requirements = state['requirements']
        current_code = state['current_code']

        # 整理之前的代码结果
        prev_code = state.get('prev_code', '')
        stdout = []
        stderr = []
        for i, res in enumerate(state["execution_result"]):
            stdout.append(res['stdout'])
            stderr.append(res['stderr'])
        # 生成错误摘要
        errorSummary = self.errorManager.summarize_errors(stderr)
        # 生成代码变更摘要
        if prev_code:
            codeDiff = self.codeManager.get_diff_summary(prev_code, current_code)
            if codeDiff['diff_lines'] > 30:
                codeDiff = prev_code
            elif not codeDiff['has_changes']:
                codeDiff = ''
            else:
                codeDiff = f'''
                新增行数: {codeDiff['added_lines_count']}
                删除行数: {codeDiff['removed_lines_count']}
                修改的函数: {' '.join(codeDiff['changed_functions'])}
                '''
        else:
            codeDiff = ''

        prompt = f'''
        你是一个Python代码调试专家，Python版本为{platform.python_version()}。分析以下错误修改代码。
        
        *说明*
        - *当前代码*为根据*建模逻辑*，使用*输入数据说明*中的字段进行数据处理与建模的代码。
        - *当前错误*为当前代码的报错，*当前结果*为当前代码的输出结果。
        - *历史代码变更*为上一版本的代码修改过的信息。
        - *历史报错*为历史版本的代码出现过的报错信息。
        
        *当前依赖*
        {requirements}
        
        *当前代码*
        {state["current_code"]}
        
        *当前错误*
        {stderr[-1]}
        
        *当前结果*
        {stdout[-1]}
        
        *历史代码变更*:
        {codeDiff}
        
        *历史报错*:
        {errorSummary}
        
        *建模逻辑*:
        {state['modeling_logic']}
        
        *输入数据说明*
        {state["data_fields"]}
        
        *要求*:
        1. 只针对代码的报错进行修改，返回修改后的依赖库文件requirements.txt和python可执行文件main.py
        2. main.py要求:
            - 确保使用sys.argv接收两个个参数: '结果csv文件输出目录'str,'图片输出目录'str
            - 添加必要的注释，不需要设置错误处理机制，让bug充分暴露。
            - 若需要保存dataframe，需确保保存的每一列都有名字(如果也要保存索引，则索引也得有名字)
            - 若*建模逻辑*里需要生成最终的文件，请将该文件放在'结果csv文件输出目录'下，并同样以csv格式保存
            - 尽量选择简单且使用人数多的库，若遇到比较复杂或频繁bug的库，需考虑手动实现该功能。
            - 路径分隔符无要求，推荐使用os.path.join
            - 必须导入包“plt.rcParams['font.sans-serif'] = ['SimHei'] plt.rcParams['axes.unicode_minus'] = False”
            - **保留原代码的print输出逻辑和结果、画图结果、sys.argv按要求正确接收参数**
        
        *JSON格式要求*
        1. 字符串必须用双引号包裹
        2. 字符串中的双引号必须转义
        3. 对象键必须用双引号包裹
        4. 每个属性后必须有逗号（除了最后一个）
        5. 布尔值必须是 true 或 false（小写）
        6. 中文字符可以直接使用，但特殊字符需要转义
        
        *返回格式*
        JSON是最外层且唯一的输出。
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:
        {{"requirements": "依赖库文件requirements", "python": "python可执行文件main", "path": ["若需要保存csv文件，保存的文件名称(csv格式)"]}}
        '''

        prompts = [
            {'role': 'system', 'content': prompt},
        ]

        class Code(BaseModel):
            requirements: str = Field(description='依赖库文件requirements.txt', default="")
            python: str = Field(description='python可执行文件main.py', default="")
            path: List[str] = Field(description='若需要保存csv文件，保存的文件名称(如xxx.csv),储存在List里', default=[])

            @field_validator('requirements')
            @classmethod
            def requirements_check(cls, r: str) -> str:
                """开头不是字母，则报错"""
                if len(r) != 0:
                    first_char = r[0]
                    # 判断是否为空白字符（空格、制表符、换行等）
                    if first_char.isspace():
                        raise ValueError('格式错误: requirements文件以空白字符起始')
                    # 判断是否为标点符号
                    if first_char in string.punctuation:
                        raise ValueError('格式错误: requirements文件以标点符号起始')
                return r

            @field_validator('python')
            @classmethod
            def python_check(cls, p: str) -> str:
                """开头不是import from"""
                if len(p) != 0:
                    first_char = p[0]
                    # 判断是否为空白字符（空格、制表符、换行等）
                    if first_char.isspace():
                        raise ValueError('格式错误: python文件格式有误，以空白字符起始，预期以import/from起始')
                    # 判断是否为标点符号
                    if first_char in string.punctuation:
                        raise ValueError('格式错误: python文件格式有误，以标点符号起始，预期以import/from起始')
                return p

        raw_response = self.llm.invoke(prompts)
        raw_content = raw_response.content
        cleaned_content = clean_json_response(raw_content)  # 清理响应
        data = json.loads(cleaned_content)
        response = Code(**data)

        return {
            "requirements": response.requirements if response.requirements else state["requirements"],
            "current_code": response.python if response.python else state["current_code"],
            "prev_code": current_code if response.python else state.get("prev_code", current_code),
            "iteration_count": state.get("iteration_count", 0) + 1,
            "paths": response.path if response.path else state["paths"],
        }

    def evaluate_result(self, state: CodeAgentState) -> CodeAgentState:
        print(f'  ===> code_step{state["step"]}: 开始校验输出')

        stdout = state["execution_result"][-1]["stdout"]
        paths = state["paths"]

        prompt = f'''
        你是一个专业的Python代码逻辑校验工具。
        *背景描述*
        当前python*代码*使用sys.argv接收了参数，运行得到*输出*{",并输出文件保存在*输出文件路径*中" if paths else ""}。
        *任务*
        请你根据*要求*判断 代码 和 输出文件路径(若存在) 是否合理，若不合理，则给出微调后的代码和输出文件路径。注意，保留代码的结构逻辑不能删减。
        
        *代码*
        {state["current_code"]}
        *输出*
        {stdout}
        *输出文件名称*
        {paths}
        *要求*
        1. 代码执行过程的每一步需指定格式print出来:
            ①使用编号"1、2..."标记步骤名称，如"1.数据读取"，"2.数据预览"
            ②说明该步骤的具体操作，如"读取了csv文件"，"使用describe进行了描述性统计"，"使用ARMA模型对字段year, gdp进行了建模"
            ③使用output标识该步骤的结果，如"output:共100个样本", "output:描述性统计的结果为...", "output:相关性热力图输出到了路径D:/data/img.jpg"
            ④每一步的完整格式为"1.相关性分析 n使用numpy对字段gdp和population进行了相关性分析，并将结果以热力图的方式可视化。output:热力图输出到了路径D:/data/img.jpg"
        2. 适当画图可视化，将图片输出到'图片输出目录'，图片以"步骤名称+图片含义"的方式命名，以jpg的格式输出。
        3. 需要使用sys.argv依次接收两个参数: '结果csv文件输出目录'str,'图片输出目录'str。
        4. 若最终生成保存了文件，请将该文件放在'结果csv文件输出目录'下，并同样以csv格式保存。
        5. 若代码需要保存dataframes，需确保保存的每一列都有名字(如果也要保存索引，则索引也得有名字)
        6. 路径分隔符无要求，推荐使用os.path.join
        7. 不检查代码计算结果的合理性（如计算的数值失真，假设检验值异常等），只要代码逻辑正确
        
        *JSON格式要求*
        1. 字符串必须用双引号包裹
        2. 字符串中的双引号必须转义
        3. 对象键必须用双引号包裹
        4. 每个属性后必须有逗号（除了最后一个）
        5. 布尔值必须是 true 或 false（小写）
        6. 中文字符可以直接使用，但特殊字符需要转义
        
        *返回格式*
        JSON是最外层且唯一的输出。
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:
        {{"valid": "代码是否符合要求，输出'符合'或'不符合'", "python": "若代码不符合要求，微调后的python代码", "path": ["若存在，代码中保存csv文件名(csv格式)"], "reason": "你的判断理由"}}
        '''

        prompts = [
            {'role': 'system', 'content': prompt},
        ]
        class Reason(BaseModel):
            valid: str = Field(description="代码是否符合要求，输出'符合'或'不符合'", default="符合")
            python: str = Field(description='若代码不符合要求，微调后的python代码', default="")
            path: List[str] = Field(description='代码中保存csv文件名(若存在),放在列表里', default=[])
            reason: str = Field(description='你的判断理由', default="")

        raw_response = self.llm.invoke(prompts)
        raw_content = raw_response.content
        cleaned_content = clean_json_response(raw_content) # 清理响应
        data = json.loads(cleaned_content)
        response = Reason(**data)

        print(f'生成文件: {paths}')
        print(f'评估原因:{response.reason}')

        return {
            "state": "complete" if response.valid == "符合" else "continue",
            "current_code": response.python if not (response.valid == "符合") else state["current_code"],
            "paths": response.path if not (response.valid == "符合") else state["paths"],
        }

    def run_with_user_interaction(
            self,
            files: List[str],
            thread_id: str,
            modeling_logic: str,
            step: int,
            max_iterations: int = 5,
    ):
        """运行图并处理用户交互"""
        initial_state = {
            "modeling_logic": modeling_logic,
            "max_iterations": max_iterations,
            "step": step,
            "files": files,
        }
        config = {
            'configurable': {
                'thread_id': thread_id,
            }
        }

        # 开始执行
        for trunk in self.graph.stream(
                initial_state,
                config,
        ):
            # print(trunk)
            # 检查是否有中断，有则调用函数输入参数
            # if "__interrupt__" in trunk:
            #     cmd = self.parameter_input()
            #     if cmd:
            #         for resume_trunk in self.graph.stream(cmd, config):
            #             # print(resume_trunk)
            #             continue
            continue

        # 返回最终代码执行结果
        state = self.graph.get_state(config).values
        x = state.get("execution_result", "")
        if x:
            return (state.get("paths", []), x[-1]['stdout'])
        else:
            return ("未返回任何结果。", "未返回任何结果。")


    def parameter_input(self):
        '''定义: Command函数根据需求传入函数参数'''
        # parameter_schema = self.graph.get_state(config).values.get("parameter_schema", {})
        response = input(f'是否准备好数据(Yes/No)(路径为{self.project_dir/"data"}):')
        response = response.strip().title()
        return Command(resume=response)


def code(model, sandbox, files, thread_id, modeling_logic, step):
    '''将CodeGenerationAgent包装成一个函数'''
    agent = CodeGenerationAgent(model, sandbox)
    out_files, outcome = agent.run_with_user_interaction(
        files=files,
        thread_id=thread_id,
        modeling_logic=modeling_logic,
        step=step
    )
    return {
        "out_files": out_files,
        "outcome": outcome,
    }


class CodeState(TypedDict):
    modeling_logic: str
    data_fields: List[Dict[str,str]]
    steps_all: int
    files: List[str]

    # 初始数据的描述性统计
    init_files: List[str]
    # 当前为第几步
    step: int
    # 规划的编程步骤
    modeling_steps: List[str]
    # 执行结果
    outcome_steps: Annotated[
        List[str],
        lambda left, right: (left or []) + (right or [])
    ]

class CodeAgent:
    def __init__(self, model, project_name, base_workspace: str="D:\\AiProgram\\Thesis"):
        # 初始化LLM
        self.llm = model
        # 初始化检查点（支持多轮对话）
        self.checkpointer = MemorySaver()
        # 构建状态图
        self.graph = self._build_graph()
        self.sandbox = CodeSandbox(base_workspace=base_workspace)
        self.project_dir = self.sandbox.create_project(project_name)

    def _build_graph(self):
        workflow = StateGraph(CodeState)
        workflow.add_node('program_node', self.program_node)
        workflow.add_node('code_node', self.code_node)

        workflow.set_entry_point("program_node")
        workflow.add_edge('program_node', 'code_node')
        workflow.add_conditional_edges(
            'code_node',
            self.route_func,
            {
                "continue": 'code_node',
                "complete": END,
            }
        )
        # 编译图
        return workflow.compile(checkpointer=self.checkpointer)

    def route_func(self, state: CodeState):
        '''判断是否还有未完成的规划'''
        steps_all = state["steps_all"]
        step = state["step"]
        if step <= steps_all:
            return "continue"
        else:
            return "complete"

    def program_node(self, state: CodeState) -> CodeState:
        print(' >> code_step: 开始规划编程')

        # 等待用户准备数据
        cont = interrupt(
            f"数据是否已经准备(Yes/No)，项目目录为{self.project_dir}?"
        )
        if cont == 'Yes':
            pass
        else:
            raise ValueError('数据未准备好！')

        # 准备好的数据文件名
        files_name = os.listdir(self.project_dir / 'data')
        files = []
        for name in files_name:
            if '.' in name:
                files.append(self.project_dir / 'data' / name)
        print(f'读取到的文件: {files}')

        data_fields = "\n".join([
            f"- {f['name']}: {f['type']} ({f['describe']})"
            for f in state.get("data_fields", [])
        ])

        prompt = f'''
        你是一个专业的Python编程专家。
        *任务*
        给定*编程需求*和*传入的数据*，其中*传入的数据*对可用的数据字段、类型、描述进行了定义，利用该数据实现去*编程需求*。你需要将*编程需求*细化拆分为可执行的多个步骤。
        *编程需求*
        {state["modeling_logic"]}
        *传入的数据*
        {data_fields}
        *要求*
        1. 将*编程需求*细化为可实现的3-5个步骤，不能出现"基于前一个步骤"，"在步骤2的基础上"这类的词。
        2. 只能使用*传入的数据*提及的数据字段，不能使用未提及的字段进行辅助数据处理或建模等。
        3. 在对数据进行处理和建模时，需明确指定数据字段名。
        4. 每个步骤结束后，需指定要保存的全部数据字段（保存为csv文件），该数据文件将作为下一步骤的传入数据，因此需保证前后步骤数据字段的完整性和一致性。
        5. 每个步骤都需要定义明确的执行内容和使用的字段名，包括使用的模型、使用到的*传入的数据*的字段名、建模方法等。
        6. 若没有更好的想法，可将步骤分为: 数据预处理、建模步骤一、建模步骤二...
        *输出格式*
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:。
        {{"plan": ["规划的3-5个编程步骤"]}}
        '''

        prompts = [
            {"role": "system", "content": prompt},
        ]
        class Program(BaseModel):
            plan: List[str] = Field(description="规划的3-5个编程步骤，以List形式存储")

        structured_model = self.llm.with_structured_output(Program)
        response = structured_model.invoke(prompts)

        print(response.plan)

        return {"modeling_steps": response.plan, "steps_all": len(response.plan), "step": 1, "files": files, "init_files": [str(f) for f in files]}

    def code_node(self, state: CodeState) -> CodeState:
        step = state["step"]
        print(f' >> code_step: 开始编程步骤{step}')
        ret = code(self.llm, self.sandbox, state["files"], str(step), state["modeling_steps"][step-1], step)

        dir_name = self.project_dir / 'data'
        files = []
        for name in ret["out_files"]:
            if '.' in name:
                files.append(dir_name / name)
        return {"files": files, "outcome_steps": [ret["outcome"]], "step": state["step"]+1}

    def run_with_user_interaction(
            self,
            modeling_logic: str,
            data_fields: List,
            thread_id: str,
    ):
        """运行图并处理用户交互"""
        initial_state = {
            "modeling_logic": modeling_logic,
            "data_fields": data_fields,
        }
        config = {
            'configurable': {
                'thread_id': thread_id,
            }
        }

        # 开始执行
        for trunk in self.graph.stream(
                initial_state,
                config,
        ):
            # print(trunk)
            # 检查是否有中断，有则调用函数输入参数
            if "__interrupt__" in trunk:
                cmd = self.parameter_input()
                if cmd:
                    for resume_trunk in self.graph.stream(cmd, config):
                        # print(resume_trunk)
                        continue
            continue

        # 返回最终代码执行结果
        state = self.graph.get_state(config).values
        outcome = state.get("outcome_steps", []) # 代码执行结果
        init_files = state.get("init_files", []) # 初始数据的路径
        modeling_steps = state.get("modeling_steps", []) # 建模详细步骤
        return {"outcome": outcome, "init_files": init_files, "modeling_steps": modeling_steps}

    def parameter_input(self):
        '''定义: Command函数根据需求传入函数参数'''
        # parameter_schema = self.graph.get_state(config).values.get("parameter_schema", {})
        response = input(f'是否准备好数据(Yes/No)(路径为{self.project_dir/"data"}):')
        response = response.strip().title()
        return Command(resume=response)




# if __name__ == '__main__':
#     from langchain_openai import ChatOpenAI
#     API_KEY = 'sk-a6647968836d4d9587d9adb77e659727'
#     # 使用langchain创建访问OpenAI的Model。
#     model = ChatOpenAI(
#         model="qwen3-coder-next",
#         openai_api_key=API_KEY,
#         openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
#         temperature=0.7,
#         response_format={"type": "json_object"}
#     )
#     agent = CodeAgent(model, "Finance")
#     result = agent.run_with_user_interaction(
#         "使用最小二乘法对数据建模，探寻GDP与人口的关系。首先对population进行正态性检验和标准化，去除空值等预处理操作。然后，建立对ggp和population建立线性模型。最后，对拟合效果进行分析，如R2、残差、残差正态性检验等。",
#         [
#             {"name": 'year', 'type': 'int', "describe": '样本年份'},
#             {"name": 'gdp', 'type': 'float', "describe": '当年国家的gdp'},
#             {"name": 'population', 'type': 'int', "describe": '当年国家的人口数量'},
#         ],
#         '1',
#     )
#     print(result)


