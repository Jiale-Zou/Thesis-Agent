import asyncio
import re
import json
import platform
from operator import add # 配合Annotated使用，将里面的元素用+合并值

from langgraph.config import get_stream_writer
from pydantic.v1 import JsonError

from CodeManager import CodeDiffManager, ErrorSummarizer, CodeSandbox

from typing import TypedDict, Annotated, List, Optional, Dict, get_args
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


class CodeAgentState(TypedDict):
    '''使用结构化数据而非自然语言描述
    - 代码历史只保留最近3个版本
    - 错误信息摘要而非完整堆栈跟踪
    - 使用add_messages自动管理对话历史
    '''

    project_name: str

    # 数据字段定义（结构化存储，减少token）
    data_fields: List[Dict[str, str]]  # [{name, type, description}]

    # 建模逻辑（精简描述）
    modeling_logic: str

    # 当前代码状态
    requirements: str
    current_code: str
    parameter_schema: Dict # 要传入的参数说明
    parameter: Dict # 交互传入的参数
    input: bool # 是否要交互传入

    # 上一轮代码状态
    prev_code: str

    # 执行结果
    execution_result: Annotated[
        List[Dict[str, str]],
        lambda left, right: (left or []) + (right or [])
    ]  # {stdout, stderr, exit_code}

    # 改进建议
    proposal: str

    # 当前运行状态
    state: str # debug, success, complete, continue

    # 迭代控制
    iteration_count: int
    max_iterations: int


class CodeGenerationAgent:
    def __init__(self, model):
        # 初始化LLM
        self.llm = model
        # 初始化检查点（支持多轮对话）
        self.checkpointer = MemorySaver()
        # 构建状态图
        self.graph = self._build_graph()
        # 构建沙盒
        self.sandbox = CodeSandbox()
        # 初始化 代码管理器 和 报错管理器
        self.codeManager = CodeDiffManager()
        self.errorManager = ErrorSummarizer()

    def _build_graph(self):
        workflow = StateGraph(CodeAgentState)

        # 添加节点
        workflow.add_node("generate_code", self.generate_code)
        workflow.add_node("params_input_node", self.params_input_node)
        workflow.add_node("execute_code", self.execute_code)
        workflow.add_node("evaluate_result", self.evaluate_result)
        workflow.add_node("debug_and_fix", self.debug_and_fix)

        # 设置入口
        workflow.set_entry_point("generate_code")

        # 添加边
        workflow.add_edge("generate_code", "params_input_node")
        workflow.add_edge("params_input_node", "execute_code")
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
                "continue": "debug_and_fix",
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

    def generate_code(self, state: CodeAgentState) -> CodeAgentState:
        """生成Python代码"""
        print(' >>> code_step: 开始生成代码。')
        writer = get_stream_writer()
        writer({'code_step': '开始生成代码。'})

        # 构建数据说明
        data_fields = "\n".join([
            f"- {f['name']}: {f['type']} ({f['describe']})"
            for f in state.get("data_fields", [])
        ])

        prompt = f"""
你是一个专业的Python代码生成器。Python版本为{platform.python_version()}。
*输入数据字段*
{data_fields}

*建模逻辑*
{state['modeling_logic']}

*要求*
1. 代码必须包含完整的错误处理
2. 添加必要的注释
3. 生成两个文件，第一个为依赖库(requirements.txt)，第二个为完整的代码(main.py)
4. 在*输出*区域内返回Python代码，不要解释。
5. 需将所有步骤以及结果用print打印出来，方便后续根据print结果判断是否满足用户需求
6. 尽量只用常见的库实现
7. 若涉及到数据处理与建模，①需确保**使用的字段只涉及*输入数据说明*中提及的字段名**②不对数据的数量/质量、模型的拟合效果进行检查

*代码格式*
**1. requirements**: 只包含库的名称。
**2. main**: 必须设置程序主入口'__main__'，并在里面调用函数。
    函数所需的参数通过外部控制台传入，其中必须的参数是文件路径，因此需设置'sys.argv'接收外部传入参数。
    如果需要画图，则也需要传入图片输出路径，并且每张图片的保存路径用其含义进行命名。
    要画图，则必须导入包“plt.rcParams['font.sans-serif'] = ['SimHei'] plt.rcParams['axes.unicode_minus'] = False”

*输出*
- 返回可执行的requirements.txt在下面的标签中
<require>
 # your requirements.txt
</require>
- 返回可执行的python代码在下面块中
<python>
 # your python
</python>
- 返回sys.argv接收的参数列表在下面块中，严格以JSON格式展示。key为参数名称，value为该参数的含义。(若不需要传入，则可以为空)
<parameter>
{{"key": "value"}}
</parameter>
"""
        prompts = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': f'按照要求在指定区域生成可执行的代码。'},
    ]

        response = self.llm.invoke(prompts).content

        # 提取requirements和python code
        # 1. 提取 requirements
        require_pattern = r"<require>(.*?)</require>"
        require_match = re.search(require_pattern, response, re.DOTALL | re.MULTILINE)
        requirements = require_match.group(1).strip() if require_match else ""
        # 2. 提取python code
        python_pattern = r"<python>(.*?)</python>"
        python_match = re.search(python_pattern, response, re.DOTALL | re.MULTILINE)
        code = python_match.group(1).strip() if python_match else ""
        # 3. 提取输入的参数
        parameter_pattern = r"<parameter>(.*?)</parameter>"
        parameter_match = re.search(parameter_pattern, response, re.DOTALL | re.MULTILINE)
        parameter = parameter_match.group(1).strip() if parameter_match else ""
        parameter = json.loads(parameter.strip())


        return {
            "data_fields": data_fields,
            "requirements": requirements,
            "current_code": code,
            "parameter_schema": parameter,
            "input": True if parameter else False,
            "iteration_count": state.get("iteration_count", 0) + 1
        }

    def params_input_node(self, state: CodeAgentState) -> CodeAgentState:
        '''交互节点，可以人为传递函数参数'''
        print(f' >>> code_step: 交互传递参数。')
        writer = get_stream_writer()
        writer({'code_step': '交互传递参数。'})

        if_input = state['input']
        parameter_schema = state['parameter_schema']
        parameter_str = '\n'.join([''.join([key, ':', value]) for key, value in parameter_schema.items()])
        if if_input:
            response = interrupt(
                f"为运行Python代码，需要传入参数：\n{parameter_str}"
            )
            return {"parameter": response}
        return {}


    def execute_code(self, state: CodeAgentState) -> CodeAgentState:
        """在沙箱中执行代码"""
        print(' >>> code_step: 开始执行代码。')
        writer = get_stream_writer()
        writer({'code_step': '开始执行代码。'})

        project_dir = self.sandbox.create_project(state.get('project_name', None))
        install_res = self.sandbox.install_dependencies(project_dir, state["requirements"])
        if install_res['success']:
            execute_res = self.sandbox.execute_code(project_dir, list(state.get("parameter", {}).values()), state["current_code"])
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

        print(f' >>> code_step: 开始调试和修复代码(Iteration {state.get("iteration_count", 0)})。')
        writer = get_stream_writer()
        writer({'code_step': f'开始调试和修复代码(Iteration {state.get("iteration_count", 0)})。'})
        print(state['execution_result'][-1]['stdout'])

        requirements = state['requirements']
        current_code = state['current_code']

        if state["state"] == "debug":
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
            # 数据说明
            data_fields = state.get("data_fields", "")
            prompt = f'''
你是一个代码调试专家。分析以下错误并提供具体的修复建议和修改后的代码。

*当前依赖*:
{requirements}

*当前代码*:
{state["current_code"]}

*当前代码sys.argv接收的参数*
{state["parameter_schema"]}

*当前错误*:
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
{data_fields}

*要求*:
1. 代码必须包含完整的错误处理
2. 添加必要的注释
3. 生成两个文件，第一个为依赖库(requirements)，第二个为完整的代码(main)
4. 在*输出*区域内返回Python代码，不要解释。
5. 需将所有步骤以及结果用print打印出来，方便后续根据print结果判断是否满足用户需求
6. 代码中只能使用*输入数据字段*中的数据字段
7. 需保证修改后的代码仍然能通过'sys.argv'接收相同的参数
8. 若代码中涉及到数据处理和建模，需确保**使用的字段只涉及*输入数据说明*中提及的字段名**
9. 尽量只用常见的库实现

*代码格式*:
**1. requirements**: 只包含库的名称。
**2. main**: 必须设置程序主入口'__main__'，并在里面调用函数。函数所需的参数通过外部控制台传入，其中必须的参数是文件路径，因此需设置'sys.argv'接收外部传入参数。

*输出*:
- 返回可执行的requirements.txt在下面的标签中
<require>
 # your requirements.txt
</require>
- 返回可执行的python代码在下面块中
<python>
 # your python
</python>
'''
        else:
            prompt = f'''
你是一个代码调试专家。分析以下错误并提供具体的修复建议和修改后的代码。


*建模逻辑*:
{state['modeling_logic']}

*输入数据说明*
{state.get("data_fields", "")}

*代码*
{state["current_code"]}

*代码sys.argv接收的参数*
{state["parameter_schema"]}

*代码输出*
{state["execution_result"][-1]["stdout"]}

*修改建议*
{state['proposal']}

*要求*:
1. 代码必须包含完整的错误处理
2. 添加必要的注释
3. 在*输出*区域内返回Python代码，不要解释。
4. 需将所有步骤以及结果用print打印出来，方便后续根据print结果判断是否满足用户需求
5. 代码中只能使用*输入数据字段*中的数据字段
6. 需保证修改后的代码仍然能通过'sys.argv'接收相同的参数
7. 若代码中涉及到数据处理和建模，需确保**使用的字段只涉及*输入数据说明*中提及的字段名**
9. 尽量只用常见的库实现

*代码格式*:
必须设置程序主入口'__main__'，并在里面调用函数。函数所需的参数通过外部控制台传入，其中一个必须的参数是文件路径，因此需设置'sys.argv'接收外部传入参数。

*输出*:
- 返回可执行的python代码在下面块中
<python>
 # your python
</python>
'''
        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': '请根据错误建议提供修正的代码。'}
        ]

        response = self.llm.invoke(prompts).content
        # 提取requirements和python code
        # 1. 提取 requirements
        require_pattern = r"<require>(.*?)</require>"
        require_match = re.search(require_pattern, response, re.DOTALL | re.MULTILINE)
        requirements = require_match.group(1).strip() if require_match else ""
        # 2. 提取python code
        python_pattern = r"<python>(.*?)</python>"
        python_match = re.search(python_pattern, response, re.DOTALL | re.MULTILINE)
        code = python_match.group(1).strip() if python_match else ""

        return {
            "requirements": requirements if requirements else state["requirements"],
            "current_code": code,
            "prev_code": current_code,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }


    def evaluate_result(self, state: CodeAgentState) -> CodeAgentState:
        """评估执行结果是否满足需求"""

        print(f' >>> code_step: 开始评估代码完整性。')
        writer = get_stream_writer()
        writer({'code_step': '开始评估代码完整性。'})

        # 如果成功，检查是否满足需求
        prompt = f'''
评估代码执行结果是否满足原始需求。

*建模步骤*:
{state['modeling_logic']}

*最终结果*
{state["execution_result"][-1]["stdout"]}

*任务*
1. *最终结果*为代码print的结果，展示了代码的执行过程。
2. 判断最终结果是否满足*建模步骤*。只需对建模步骤的完整性进行评估，即检查建模逻辑中的步骤是否都有相应的输出结果，而不需对运行结果的合理性和建模结果的好坏进行评估（哪怕结果不合理，但只要步骤完整，也认为满足）。
3. 最后输出一个JSON格式，包含两项:
- valid: Yes/No(建模步骤是否完整) 
- reason: 简述原因

*输出*: 严格遵循JSON的格式，不要有任何其他内容和其他形式的输出，输出的格式如下。
{{"valid": "", "reason": ""}}
'''
        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': '请严格遵守要求输出结果。'}
        ]

        response = self.llm.invoke(prompts).content
        json_response = {"valid": "no", "reason": ""}
        try:
            j = json.loads(response)
            for key, value in j.items():
                json_response[key.lower()] = value.lower()
        except Exception as e:
            raise JsonError(f'plan node "JSON loads error": {e}:')

        print(f' >>> code_step: 代码评估结果为: {json_response["valid"]}。原因: {json_response["reason"]}')
        writer({'code_step': f'代码评估结果为: {json_response["valid"]}。原因: {json_response["reason"]}'})

        if json_response["valid"] == "yes":
            return {
                "state": "complete",
            }
        else:
            return {
                "state": "continue",
                "proposal": json_response["reason"],
            }

    def run_with_user_interaction(
            self,
            thread_id: str,
            project_name: str,
            data_fields: List[Dict[str, str]],
            modeling_logic: str,
            max_iterations: int = 5,
    ):
        """运行图并处理用户交互"""
        initial_state = {
            "project_name": project_name,
            "data_fields": data_fields,
            "modeling_logic": modeling_logic,
            "max_iterations": max_iterations,
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
                cmd = self.parameter_input(config)
                if cmd:
                    for resume_trunk in self.graph.stream(cmd, config):
                        # print(resume_trunk)
                        continue

        # 返回最终代码执行结果
        x = self.graph.get_state(config).values.get("execution_result", "")
        if x:
            return x[-1]['stdout']
        else:
            return "未返回任何结果。"


    def parameter_input(self, config):
        '''定义: Command函数根据需求传入函数参数'''
        parameter_schema = self.graph.get_state(config).values.get("parameter_schema", {})
        if not parameter_schema:
            return

        response = {}
        for key, value in parameter_schema.items():
            response[key] = input(f'{key}({value}):')
        return Command(resume=response)




if __name__ == '__main__':
    from langchain_openai import ChatOpenAI
    API_KEY = 'sk-a6647968836d4d9587d9adb77e659727'
    # 使用langchain创建访问OpenAI的Model。
    model = ChatOpenAI(
        model="qwen3-coder-next",
        openai_api_key=API_KEY,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7
    )
    agent = CodeGenerationAgent(model)
    result = agent.run_with_user_interaction(
        '1',
        'Finance',
        [{'name': 'country_id', 'type': 'string', 'describe': '国家ISO三位代码，用于标识观测单位'}, {'name': 'year', 'type': 'integer', 'describe': '年份，范围为1960–2022'}, {'name': 'gdp_growth', 'type': 'float', 'describe': '实际GDP年增长率（%），来源于World Bank WDI'}, {'name': 'gdp_per_capita_growth', 'type': 'float', 'describe': '实际人均GDP年增长率（%），来源于World Bank WDI'}, {'name': 'pop_total', 'type': 'float', 'describe': '总人口（百万），来源于UN World Population Prospects'}, {'name': 'pop_15_64', 'type': 'float', 'describe': '15–64岁人口（百万），来源于UN World Population Prospects'}, {'name': 'pop_65plus', 'type': 'float', 'describe': '65岁及以上人口（百万），来源于UN World Population Prospects'}, {'name': 'pop_0_14', 'type': 'float', 'describe': '0–14岁人口（百万），来源于UN World Population Prospects'}, {'name': 'life_expectancy', 'type': 'float', 'describe': '出生时预期寿命（岁），来源于IHME GBD'}, {'name': 'schooling_mean', 'type': 'float', 'describe': '25岁以上人口平均受教育年限（年），来源于Barro-Lee dataset'}, {'name': 'invest_ratio', 'type': 'float', 'describe': '国内总投资占GDP比重（%），来源于World Bank WDI'}, {'name': 'gov_consumption', 'type': 'float', 'describe': '政府最终消费支出占GDP比重（%），来源于World Bank WDI'}, {'name': 'trade_openness', 'type': 'float', 'describe': '进出口总额占GDP比重（%），来源于World Bank WDI'}, {'name': 'support_ratio', 'type': 'float', 'describe': '支持比，计算式为 pop_15_64 / (pop_0_14 * 0.3 + pop_65plus * 0.8 + pop_15_64 * 1.0)，其中0.3、0.8、1.0为UN Age-Specific Consumption Weights'}, {'name': 'old_dependency_ratio', 'type': 'float', 'describe': '老年抚养比，计算式为 pop_65plus / pop_15_64'}, {'name': 'young_dependency_ratio', 'type': 'float', 'describe': '少儿抚养比，计算式为 pop_0_14 / pop_15_64'}, {'name': 'human_capital_density', 'type': 'float', 'describe': '人力资本密度，计算式为 schooling_mean * (pop_15_64 / pop_total)'}, {'name': 'log_pop_total', 'type': 'float', 'describe': '总人口自然对数，即 log(pop_total)'}],
        '采用两阶段估计策略：第一阶段使用系统广义矩估计（System GMM）估计动态面板模型 gdp_growth_it = α + β1·gdp_growth_i,t−1 + β2·support_ratio_it + β3·old_dependency_ratio_it + β4·log_pop_total_it + β5·human_capital_density_it + γ·X_it + μ_i + ε_it，其中X_it为invest_ratio、gov_consumption、trade_openness等控制变量，μ_i为国家固定效应，ε_it为扰动项；工具变量集包括所有解释变量的二阶及更高阶滞后项（差分形式）与被解释变量的滞后二阶差分；第二阶段对old_dependency_ratio进行门槛回归，以gdp_growth为因变量，以old_dependency_ratio为门槛变量，识别其对GDP增长影响发生结构性突变的临界值，并检验门槛效应的显著性（Bootstrap法）。所有模型均在Python中通过linearmodels库实现系统GMM估计，通过thresholdmodels库实现门槛回归；数据清洗与特征工程（如support_ratio、human_capital_density、log_pop_total的构造）使用pandas完成，缺失值采用多重插补（fancyimpute）处理。'
    )
    print(result)


