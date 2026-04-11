import os
import re
import subprocess
import warnings

import pandas as pd
import pypandoc
import requests
import json
import uuid
import asyncio
from typing import TypedDict, List, Optional, Dict, Any, Annotated, Literal

from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.config import get_stream_writer
from langchain_core.messages import AnyMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from Thesis.BlackBox import CodeAgent, clean_json_response
from ThesisAgent.BlackBox import CodeGenerationAgent
from Model import chat_model, code_model, parser

warnings.filterwarnings('ignore')

''' 论文写作Agent
工作流：规划 → 检索 → 解析 → 开题报告 → 撰写 → 评审
'''

invalid_chars = r'[\\/:*?"<>|#]'  # 不计入字数统计的符号

@tool("get_description")
def get_description(paths: List[str]) -> str:
    """
    根据传入的文件路径，对文件进行描述性统计，并将结果生成Markdown格式表格
    Args:
        paths (List[str]): 传入的文件路径，以List形式存储多个需要分析的文件路径
    Returns:
        str: Markdown格式的描述性统计表
    """
    ret_tables = []
    for path in paths:
        end = path.split('.')[-1]
        if end == 'csv':
            try:
                file = pd.read_csv(path, header=0, encoding='utf-8')
            except:
                try:
                    file = pd.read_csv(path, header=0, encoding='gbk')
                except:
                    raise ValueError(f'{path}文件损坏，打开失败。')
        elif end == 'xlsx':
            try:
                file = pd.read_excel(path, header=0, engine='xlrd')
            except:
                try:
                    file = pd.read_excel(path, header=0, engine='openpyxl')
                except:
                    raise ValueError(f'{path}文件损坏，打开失败。')
        else:
            raise ValueError(f'预取文件格式为csv/xlsx，实际为{end}。')
        columns = file.columns
        dtypes = file.dtypes
        describes = file.describe()

        table = ["| 变量 | count | mean | std | min | max | nunique |", "|-----|------|------|------|------|------|-------|"]
        for col in columns:
            desc = describes[col]
            if dtypes[col] != 'object':
                table.append(
                    f"| {col} | {int(desc['count'])} | {desc['mean']:.03f} | {desc['std']:.03f} | {desc['min']:.03f} | {desc['max']:.03f} | - |"
                )
            else:
                table.append(
                    f"| {col} | {int(desc['count'])} | - | - | - | - | {int(file[col].nunique())} |"
                )
        table = '\n'.join(table)
        ret_tables.append(table)
    ret_tables = '\n\n'.join(ret_tables)

    return ret_tables





class PaperWritingState(TypedDict):
    '''
    定义全局状态State的属性
    '''
    user_input: str

    # 关键字阶段
    search_queries: List[str]  # 生成的检索关键词

    # 检索阶段
    papers_meta: List  # 解析后的论文内容（作者、年份、摘要、url）

    # 文献综述
    literature_review: str  # 文献综述
    literature_review_check: List[str] # 文献综述合规检查
    review_iteration: int
    max_review_iteration: int

    # 开题报告阶段
    theme: str
    data: List
    method: str

    # 代码执行结果
    code_result: str

    # 行文大纲部分
    thing_write_part: str
    # 指标说明部分
    indicator_part: str
    # 方法说明部分
    method_part: str
    # 抽取的表格
    chart: List[Dict[str, str]]
    # 实验结果部分
    result_part: str
    # 引言说明部分
    introduction_part: str
    # 结论说明部分
    conclusion_part: str
    # 参考文献部分
    reference_part: str

    # 论文成果
    thesis: str

    next_node: str  # 当前节点

class SupAgent:

    def __init__(self):
        ### 任务配置 ###
        config =  json.load(open('config.json', 'r'))
        self.chat = chat_model(config['chat_model']) # chat模型
        self.code = code_model(config['code_model']) # code模型
        self.PATH = config['workbase'] # 工作目录
        self.project_name = config['project_name'] # 项目名称
        self.project_dir = os.path.join(self.PATH, self.project_name)
        self.wkhtmltopdf_path = config["wkhtmltopdf_path"] # wkhtmltopdf路径
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(PaperWritingState)
        # 添加节点
        workflow.add_node('keyword_node', self.keyword_node)
        workflow.add_node('search_node', self.search_node)
        workflow.add_node('literature_node', self.literature_node)
        workflow.add_node('literature_check_node', self.literature_check_node)
        workflow.add_node('plan_node', self.plan_node)
        workflow.add_node('plan_check_node', self.plan_check_node)
        workflow.add_node('code_node', self.code_node)
        workflow.add_node('thing_write_node', self.thing_write_node)
        workflow.add_node('indicator_write_node', self.indicator_write_node)
        workflow.add_node('method_write_node', self.method_write_node)
        workflow.add_node('method_write_check_node', self.method_write_check_node)
        workflow.add_node('chart_extract_node', self.chart_extract_node)
        workflow.add_node('result_write_node', self.result_write_node)
        workflow.add_node('introduction_conclusion_write_node', self.introduction_conclusion_write_node)
        workflow.add_node('reference_node', self.reference_node)
        workflow.add_node('assemble_node', self.assemble_node)
        workflow.add_node('format_node', self.format_node)
        # 添加边
        workflow.set_entry_point("keyword_node")
        workflow.add_edge('keyword_node', 'search_node')
        workflow.add_edge('search_node', 'literature_node')
        workflow.add_edge('literature_node', 'literature_check_node')
        workflow.add_conditional_edges(
            'literature_check_node',
            self._literature_check_route,
            {
                'literature': 'literature_node',
                'plan': 'plan_node'
            }
        )
        workflow.add_edge('plan_node', 'plan_check_node')
        workflow.add_edge('plan_check_node', 'code_node')
        workflow.add_edge('code_node', 'thing_write_node')
        workflow.add_edge('thing_write_node', 'indicator_write_node')
        workflow.add_edge('indicator_write_node', 'method_write_node')
        workflow.add_edge('method_write_node', 'method_write_check_node')
        workflow.add_edge('method_write_check_node', 'chart_extract_node')
        workflow.add_edge('chart_extract_node', 'result_write_node')
        workflow.add_edge('result_write_node', 'introduction_conclusion_write_node')
        workflow.add_edge('introduction_conclusion_write_node', 'reference_node')
        workflow.add_edge('reference_node', 'assemble_node')
        workflow.add_edge('assemble_node', 'format_node')
        # 编译图
        return workflow.compile(checkpointer=self.checkpointer)


    def _literature_check_route(self, state: PaperWritingState):
        '''边函数'''
        return state["next_node"]

    def keyword_node(self, state: PaperWritingState):
        print(">>> 关键字节点")

        prompt = '''
        你是一个经济管理领域的论文写作专家。

        *任务*
        根据用户的要求，生成该论文有关的'检索关键词'，要求全面涵盖主题，最多为4个。

        *分析示例*
        **用户输入**: "我要写一篇有关ESG与股票风险关系的论文，具体来说，需要先采用某种风险测度衡量股票风险，然后选取某些指标来衡量股票ESG水平，最后再实证ESG水平与股票风险测度的关系。"
        **分析过程**:
            1. **明确主题**: 用户的主题为'ESG与股票风险'，其中涉及到两个关键字'ESG'和'股票分析'。
            2. **分析用户具体要求**: 用户提出了具体的要求，即分为三方面考虑，一是定义'风险测度'来衡量股票风险，二是选取指标衡量'ESG水平'，三是研究'相关关系'。
            3. **确定文章检索的关键字**: 根据对用户意图的分析，找到有关的的关键字为'ESG评分'、'ESG指标'、'股票风险测度'、'VaR'、'ES'、'相关性研究'、'因果关系'等。
        
        *返回结果*
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:
        {"keyword": ["检索关键词"]}
        '''
        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': state['user_input']},
        ]

        class KeyWord(BaseModel):
            keyword: List[str] = Field(description='检索关键词')

        structured_model = self.chat.with_structured_output(KeyWord)
        response = structured_model.invoke(prompts)

        print(f'plan_step: 检索关键词为{response.keyword[:4]}')

        return {'search_queries': response.keyword[:4], 'next_node': 'search'}

    def search_node(self, state: PaperWritingState):
        print("\n>>> 检索节点")

        class Meta(BaseModel):
            author: str = Field(description='作者')
            public: str = Field(description='出版社')
            year: int = Field(description='年份', ge=1900)
            abstract: str = Field(description='摘要', max_length=300)
            url: str = Field(description='论文链接')
            title: str = Field(description='标题')

            @field_validator('abstract', mode='before')
            @classmethod
            def truncate_abstract(cls, v: str) -> str:
                """截断abstract字段，如果超过2个字符"""
                if isinstance(v, str):
                    return v[:300]
                return v

        queries = state['search_queries']
        def query_parse(resp):
            ret = []
            for s in resp:
                json_s = json.loads(s['text'])
                try:
                    title = json_s['Title'].replace("[CITATION][C]", "").replace("[PDF]","").strip()
                except:
                    continue
                try:
                    url = json_s['URL'].strip()
                except:
                    url = 'http:\\mcp_search'
                try:
                    abstract = json_s['Abstract'].strip()
                except:
                    abstract = ''
                if not abstract:
                    continue
                try:
                    author, public = json_s['Authors'].split('-')[:2]
                except:
                    author = 'no_author'
                    public = 'no_public'
                author = author.strip().strip(',').replace("，", ",")
                author = ' & '.join([x.strip() for x in author.split(',')])
                year = 2000
                year_match = re.search(r'(\d{4})', public)
                if year_match:
                    year = int(year_match.group(1))
                public = re.sub(r'[^a-zA-Z\s.]', '', public).strip()
                ret.append(Meta(author=author, public=public, year=year, url=url, title=title, abstract=abstract))
            return ret

        mcp_services = json.load(open("mcp_config.json", 'r'))['mcpServers']
        client = MultiServerMCPClient(
            {
                "google-scholar": mcp_services["google-scholar"]
            }
        )
        tools = asyncio.run(client.get_tools())
        search_google_scholar_key_words = [tool for tool in tools if tool.name == 'search_google_scholar_key_words'][0]
        query_result = []
        for query in queries:
            resp = asyncio.run(search_google_scholar_key_words.ainvoke(
                {
                    'query': query,
                    'num_results': 10
                }
            ))
            query_result.extend(query_parse(resp))
        print(f'search_step: 共{len(query_result)}篇文献')
        print(query_result)

        return {"papers_meta": query_result}

    def literature_node(self, state: PaperWritingState):
        print("\n>>> 文献综述节点")

        papers_meta = state['papers_meta']
        str_papers_meta = []
        i = 1
        for meta in papers_meta:
            str_papers_meta.append(
                f'【文献{i}】标题:{meta.title} | 作者:{meta.author} | 年份:{meta.year} | 摘要:{meta.abstract}'
            )
        str_papers_meta = '\n'.join(str_papers_meta)

        supply_prompt = ''
        if state['next_node'] == 'literature':
            supply_prompt = f'''
            *文献综述初稿*
            {state['literature_review']}
            *初稿不合规*
            {state['literature_review_check']}
            '''


        prompt = f'''
        你是一个论文阅读分析专家。

        *任务*
        '根据*用户写作要求*和给定的*参考文献*信息，{'写出文献综述' if state['next_node']=='literature' else '参照*初稿不合规*，修改*参考文献*'}。
            - **只能使用给定的参考文献，不能捏造或使用未给定的参考文献**
            - **字数限制在3000-4000之间**
            - 严格遵守*注意事项*
            
        *参考文献*
        {str_papers_meta}
        
        *用户写作要求*
        {state["user_input"]}
        
        {supply_prompt}
        
        *注意事项*
        1. 引用参考文献中中的观点，必须在句末严格按照"(author, year)"的形式标注出来，且author必须按照参考文献中给定的author写法展示。
            - 给定参考文献: 【文献1】标题:人工只能在教育中的应用 | 作者:Jack & John & Kimi | 年份:2020 | 摘要:本文结合当今热点，详细论述了人工智能时代下教育的改革方向为智教结合...
            - 正确示范: John(2020)通过研究发现，人工智能时代的教育新方向为智教结合(Jack & John & Kimi, 2020)
            - 错误示范: ....(Jack, 2020) <- 遗漏
            - 错误示范: ....(Jack & john, 2020) <- 遗漏
            - 错误示范: ....(Jack et al., 2020) <- 参考文献为'Jack & John & Kimi'，但这里写为'Jack et al.'
            - 错误示范: ....(Jack, john & Kimi, 2020) <- 参考文献为'Jack & John & Kimi'，但这里写为'Jack, john & Kimi'
        2. 可分段撰写（分主题综述：理论/实证/缺口），正文段首用"&emsp;&emsp;"实现首行缩减，段尾用两个空字符"\\n\\n"实现换行
            
        *返回结果*
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:
        {{"literature": "符合格式要求，且字数为3000-4000之间的文献综述"}}
        '''
        prompts = [
                {'role': 'user', 'content': prompt}
            ]

        class Literature(BaseModel):
            literature: str = Field(description='符合格式要求，且字数为3000-4000之间的文献综述')

        structured_model = self.chat.with_structured_output(Literature)
        response = structured_model.invoke(prompts)
        print(f'文献综述:\n{response.literature}')

        return {'literature_review': response.literature, 'next_node': 'literature_check', 'review_iteration': state.get('review_iteration',0)+1}

    def literature_check_node(self, state: PaperWritingState):
        print("\n>>> 文献综述检查节点")

        def word_check(text: str):
            length = len(text)
            if 2900 <= length <= 4500:
                return ''
            return f'- 文献综述要求字数为3000-4000，实际为{length}字，请{"增加" if 2900 > length else "缩减"}文献综述的字数。'

        def review_check(text: str):
            errors = []
            words = word_check(text)
            if words:
                errors.append(words)
            pattern1 = re.compile(r"\((?P<author>.[^,.!，。！]{2,30}),\s(?P<year>\d{4})\)", re.UNICODE)
            pattern2 = re.compile(r"[，。](?P<author>.[^,.!，。！]{2,10})\((?P<year>\d{4})\)")
            for p in (pattern1, pattern2):
                for match in p.finditer(text):
                    author = match.group("author")
                    year = match.group("year")
                    for meta in state['papers_meta']:
                        if author == meta.author and str(meta.year) == year:
                            break
                    else:
                        errors.append(f'- 索引了author="{author}",year="{year}"，与参考文献不符，请检查是否存在错误或捏造参考文献。')
            return '\n'.join(errors)

        errors = review_check(state['literature_review'])
        print(f'literature_check_step: {errors}')
        if (state["max_review_iteration"] > state["review_iteration"]) and errors:
            next_node = 'literature'
        else:
            next_node = 'plan'
        return {'literature_review_check': errors, 'next_node': next_node}

    def plan_node(self, state: PaperWritingState):
        print("\n>>> 规划节点")

        prompt = '''
        你是一个经济管理领域的论文写作专家。

        *任务*
        给定*用户要求*, *文献综述*，生成论文的题目和研究方法。
        
        *要求*
        1. 研究主题(theme): 参照*文献综述*中以往学者的研究成果和*用户要求*，考虑到创新性和可行性，确定论文的题目。
        2. 研究数据(data): 明确描述研究该主题所需用到的全部数据字段，包含字段名称、字段类型、字段描述三部分。请注意，只要在后续编程过程中可能会涉及到的字段，都需要说明，包括一些辅助字段(时间、标识、过滤、聚合等)，比如'年份', 'id'等等。使用的数据字段不超过5个。
        3. 研究方法(method): 对于step1的研究主题和step2的研究数据，选取合适的方法对其进行建模，详细地描述建模过程。
            - 只能使用'data'中定义的字段名。
            - 若需要使用'data'中的字段衍生出的其他字段名（如log, exp变换），需要明确交代该衍生字段与原始字段的计算关系式。
            - 不能涉及到对研究数据的数量、质量、时间的要求。
        
        *返回结果*
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:
        {"theme": "论文题目", "method": "研究方法", "data": [{"name": "字段名称，用英文表示", "type": ""字段类型，如int,str,float等}, "description": "字段描述，对字段的含义进行说明"]}
        '''
        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'*用户要求*: "{state["user_input"]}"。*文献综述*: "{state["literature_review"]}"'},
        ]

        class Data(BaseModel):
            name: str = Field('字段名称，用英文表示')
            type: str = Field('字段类型，如int,str,float等')
            description: str = Field('字段描述，对字段的含义进行说明')

        class Structure(BaseModel):
            theme: str = Field('论文题目')
            data: List[Data] = Field('研究数据')
            method: str = Field('研究方法')

        structured_model = self.chat.with_structured_output(Structure)
        response = structured_model.invoke(prompts)

        data_str = []
        for d in response.data:
            data_str.append(
                f'- 字段名称:{d.name}({d.type}), 字段含义:{d.description}。'
            )
        data_str = '\n'.join(data_str)
        print(f'plan_step: 题目为 “{response.theme}”')
        print(f'plan_step: 数据字段为 “{data_str}”')
        return {
            "theme": response.theme,
            "data": response.data,
            "method": response.method,
            'next_node': 'plan_check'
        }

    def plan_check_node(self, state: PaperWritingState):
        print(">>> 规划检查节点")

        data = state["data"]
        data_str = []
        for d in data:
            data_str.append(
                f'- 字段名称:{d.name}({d.type}), 字段含义:{d.description}。'
            )
        data_str = '\n'.join(data_str)

        prompt = '''
        你是一个数学建模的规划专家。

        *任务*
        给定*研究方法*, *数据字段*，判断*数据字段*和*研究方法*是否满足数据要求。

        *重点关注*
        - *研究方法*中使用的数据字段是否在*数据字段*中，或者可由*数据字段*计算变换得到。
        - *研究方法*中可能隐含了数据处理和建模中所需的辅助字段(比如时间,编号,组别)，请从编程和建模的角度考虑这些必须的字段，以免实际编程时缺失。
        - 后续编程使用的工具为Python，需估计*研究方法*是否可行。若不可行，需简化为一种可行的研究方法。
        - 研究数据(data)是否不超过5个
        
        *输出*
        是否合规(valid): Yes/No(Yes表示数据字段和研究方法满足数据要求)
        研究数据(data): (valid为Yes时为空)满足要求的数据字段，明确描述研究该主题所需用到的全部数据字段，包含字段名称、字段类型、字段描述三部分
        研究方法(method): (valid为Yes时为空)满足要求的研究方法
        
        *返回结果*
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:
        {"theme": "论文题目", "method": "研究方法", "data": [{"name": "字段名称，用英文表示", "type": ""字段类型，如int,str,float等}, "description": "字段描述，对字段的含义进行说明"]}
        '''
        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'*研究方法*\n"{state["method"]}"\n*数据字段*\n"{data_str}"。请按要求进行判断'},
        ]

        class Data(BaseModel):
            name: str = Field('字段名称，用英文表示')
            type: str = Field('字段类型，如int,str,float等')
            description: str = Field('字段描述，对字段的含义进行说明')

        class Structure(BaseModel):
            valid: Literal["Yes", "No"] = Field('是否合规，只能为Yes或No')
            data: List[Data] = Field('研究数据')
            method: str = Field('研究方法')

        structured_model = self.chat.with_structured_output(Structure)
        response = structured_model.invoke(prompts)

        if response.valid.lower() == "yes":
            return {'next_node': 'code'}
        elif response.valid.lower() ==  "no":
            data_str = []
            for d in response.data:
                data_str.append(
                    f'- 字段名称:{d.name}({d.type}), 字段含义:{d.description}。'
                )
            data_str = '\n'.join(data_str)
            print(f'plan_check_step: 数据字段更改为 “{data_str}”')
            return {"data": response.data, "method": response.method, 'next_node': 'code'}
        else:
            print(f'plan_check_step: Errors(预期valid只能为Yes或No，实际为{response.valid})')
            raise ValueError(f'预期valid只能为Yes或No，实际为{response.valid}')

    def code_node(self, state: PaperWritingState):
        print("\n>>> 代码节点")

        method = state['method']
        data = state['data']
        data_dict = []
        for d in data:
            data_dict.append(
                {'name': d.name, 'type': d.type, 'describe': d.description}
            )

        agent = CodeAgent(self.code, self.project_name)
        result = agent.run_with_user_interaction(
            method,
            data_dict,
            uuid.uuid4().hex,
        ) # {}

        return {"code_result": result}

    def thing_write_node(self, state: PaperWritingState):
        print('>>> 开始撰写行文思路')

        map = {
            0: '一',
            1: '二',
            2: '三',
            3: '四',
            4: '五',
            5: '六',
            6: '七',
            7: '八',
            8: '九',
        }
        # 研究方法
        modeling_steps = state["code_result"]["modeling_steps"]
        method_detail = []
        for i, step in enumerate(modeling_steps):
            method_detail.append(
                f'- 步骤{map[i]}: {step}'
            )
        method_detail = '\n'.join(method_detail)
        # LLM交互
        prompt = f'''
        *研究方法*
        {method_detail}
        
        *任务*
        给定*研究任务*，你需要将其抽取为几个主题。主题需要使用经管、统计领域的专业名词进行概括（每个主题不超过10个字）。
        示例: 基准回归分析，非线性效应检验，稳健性检验
        
        *返回结果*
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:
        ["抽取的主题，每个主题不超过10个字"]
        '''
        prompts = [
            {'role': 'system', 'content': prompt},
        ]
        raw_response = self.chat.invoke(prompts)
        raw_content = raw_response.content
        cleaned_content = clean_json_response(raw_content)  # 清理响应
        data = json.loads(cleaned_content)
        # 生成行文大纲图片
        def generate_mermaid_png(study_node, output_file='diagram.png'):
            """
            通过 mermaid-cli 生成 PNG
            """
            a1 = []
            a2 = []
            for i, n in enumerate(study_node):
                a1.append(
                    f'D --> D{i + 1}[{n}]'
                )
                if i > 0:
                    a2.append(
                        f'    D{i + 1} --> E'
                    )
            a1 = '\n    '.join(a1)
            a2 = '\n    '.join(a2)
            mermaid_code = f"""
            graph TD
                A[研究起点] --> B[理论框架构建<br/>文献综述与理论分析]
                B --> C[研究设计<br/>模型设定、变量与数据]
                C --> D[实证分析]

                {a1}

                D1 --> E[结果整合与讨论]
                {a2}

                E --> F[研究结论与政策启示]
            """

            # 1. 创建临时文件
            temp_file = 'temp_mermaid.mmd'
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)

            # 2. 调用 mmdc 命令
            mmdc_path = json.load(open('config.json', 'r'))['mmdc_path']
            cmd = [
                mmdc_path,
                '-i', temp_file,
                '-o', output_file,
                '-t', 'forest',  # 主题：forest, dark, neutral, default
                '-b', 'white',  # 背景色
                '-s', '2'  # 缩放比例
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"图表已生成: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"生成失败: {e}")
            except FileNotFoundError:
                print("未找到 mmdc，请先安装: npm install -g @mermaid-js/mermaid-cli")
            finally:
                # 清理临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        jpg_path = os.path.join(self.project_dir, 'logs')
        jpg_path = os.path.join(jpg_path, 'research_flow.png')
        generate_mermaid_png(data, jpg_path)
        # 形成Markdown
        thing_write = f'''&emsp;&emsp;第一章为引言，介绍了研究背景和研究的意义。\\n\\n&emsp;&emsp;第二章为行文思路介绍，介绍了本文的行文结构并可视化。\\n\\n&emsp;&emsp;第三章为文献综述，对以往学者的理论和研究成功进行了梳理。\\n\\n&emsp;&emsp;第四章为数据指标，对使用的数据字段含义及其来源进行了说明。\\n\\n&emsp;&emsp;第五章为研究方法，介绍了研究方法的具体实践过程。\\n\\n&emsp;&emsp;第六章为实验结果，使用了研究方法对数据进行研究，分为{'、'.join(data)}这{len(data)}个研究层面。\\n\\n&emsp;&emsp;第七章为结论和展望，概括本研究的主要实证发现，并对未来可拓展的研究方向进行展望。\n\n![]({jpg_path})  &emsp;&emsp;*图: 行文思路流程图*'''

        return {"thing_write_part": thing_write}


    def indicator_write_node(self, state: PaperWritingState):
        print("\n>>> 数据指标撰写节点")

        method = state['method']
        data = state["data"]
        data_str = []
        for d in data:
            data_str.append(
                f'- 字段名称:{d.name}, 字段含义:{d.description}。'
            )
        data_str = '\n'.join(data_str)
        init_files = state["code_result"]["init_files"]
        prompt = f'''
        你是一个论文写作专家，现在你正在撰写论文的指标说明部分。
        
        *背景说明*
        给定的*数据说明*表示数据字段的含义，这些数据字段将被用于*研究方法*的实施。你需要以给定Mardown的格式撰写指标说明，字数为2000-2500。
        
        *数据文件路径*
        {init_files}
        *数据说明*
        {data_str}
        *研究方法*
        {method}
        
        *撰写格式*
        - ### 作为小标题标志
        - 用1/2/3...作为小标题序号
        - 不用包含”指标说明“大标题
        - 正文段首用"&emsp;&emsp;"实现首行缩减，段尾用"\\n\\n"实现换行
        - 不要使用**给正文加斜体
        - 示例:
        ### 1. 指标定义
        对研究中涉及的核心指标进行定义、解释、来源说明，使用“无序列表分条列点”说明。
        ### 2. 指标描述性统计
        对核心数据字段进行描述性统计，必须以“表格+文字”的形式展示，表格格式为：
        | 变量 | count | mean | std | min | max | nunique |
        ### 3. 指标合理性说明
        对指标选取的合理性和适用性进行说明。
        
        *输出要求*
        请严格按照*撰写格式*输出对应的Markdown格式，不要包含任何其他文本、解释或额外的格式。
        '''

        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': '请严格按照要求生成Markdown格式的指标说明文本。'}
        ]

        agent = create_react_agent(
            model = self.chat,
            tools=[get_description]
        )
        response = agent.invoke({"messages": prompts})

        return {"indicator_part": response["messages"][-1].content}

    def method_write_node(self, state: PaperWritingState):
        print("\n>>> 方法说明撰写节点")

        map = {
            0: '一',
            1: '二',
            2: '三',
            3: '四',
            4: '五',
            5: '六',
            6: '七',
            7: '八',
            8: '九',
        }
        method = state['method']
        # 实验详细步骤
        modeling_steps = state["code_result"]["modeling_steps"]
        method_detail = []
        for i, step in enumerate(modeling_steps):
            method_detail.append(
                f'- 步骤{map[i]}: {step}'
            )
        method_detail = '\n'.join(method_detail)
        prompt = f'''
        你是一个论文写作专家，现在你正在撰写论文的方法说明部分。

        *背景说明*
        给定的研究方法和研究方法的详细执行步骤，你需要以给定Mardown的格式撰写方法说明，字数为2500-3500字。
        *研究方法*
        {method}
        *研究方法详细步骤*
        {method_detail}
        
        *数学公式格式*
        1. 行内公式: $公式内容$
        2. 独立公式: $$公式内容$$
        3. 上下标: x^2, a_n, x^{{n+m}}
        4. 分式: \\frac{{a}}{{b}}
        5. 根号: \sqrt{{x}}, \sqrt[n]{{x}}
        6. 希腊字母: \\alpha, \\beta, \Omega, \pi
        7. 求和/积分: \sum_{{i=1}}^n, \int_a^b
        8. 矩阵: \\begin{{matrix}} 1 & 2 \\\\ 3 & 4 \end{{matrix}}
        9. 独立公式换行: \\begin{{aligned}} 公式 \\end{{aligned}}，公式内用在行尾用\\\\换行

        *撰写格式*
        - ### 作为小标题标志
        - 用1/2/3...作为小标题序号
        - 不用包含”方法说明“大标题
        - 对于每一小标题的明细内容，可使用无序编号(*)分条列点
        - 正文段首用"&emsp;&emsp;"实现首行缩减，段尾用两个空字符"\\n\\n"实现换行
        - 可适当插入公式进行说明，公式的格式需严格遵守*数学公式格式*。若公式过长，需对公式换行避免溢出
        - 不要使用**给正文加斜体
        - 示例:
        ### 1. 研究方法选择
        阐述研究所采用的主要方法及其适用性。
        ### 2. 模型构建
        详细描述理论模型或分析框架的构建过程。
        ### 3. 数据处理与分析技术
        说明数据收集、清洗、处理及分析的具体技术方法。

        *输出要求*
        请严格按照*撰写格式*输出对应的Markdown格式，不要包含任何其他文本、解释或额外的格式。
        '''

        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': '请严格按照要求生成Markdown格式的方法说明文本。'},
        ]

        response = self.chat.invoke(prompts).content
        return {"method_part": response}

    def method_write_check_node(self, state: PaperWritingState):
        print("\n>>> 公式检查节点")
        prompt = f'''
        你是一个Markdown公式检查专家。
        
        *任务*
        对传入markdown文本，**不改变其内容和格式**，只检查和更正其公式的正确性，返回更正后的文本。
        
        *传入markdown文本*
        {state["method_part"]}
        
        *数学公式格式*
        1. 行内公式: $公式内容$
        2. 独立公式: $$公式内容$$
        3. 上下标: x^2, a_n, x^{{n+m}}
        4. 分式: \\frac{{a}}{{b}}
        5. 根号: \sqrt{{x}}, \sqrt[n]{{x}}
        6. 希腊字母: \\alpha, \\beta, \Omega, \pi
        7. 求和/积分: \sum_{{i=1}}^n, \int_a^b
        8. 矩阵: \\begin{{matrix}} 1 & 2 \\\\ 3 & 4 \end{{matrix}}
        9. 独立公式换行: \\begin{{aligned}} 公式 \\end{{aligned}}，公式内用在行尾用\\\\换行
        
        *markdown格式*
        - ### 作为小标题标志
        - 用1/2/3...作为小标题序号
        - 对于每一小标题的明细内容，可使用无序编号(*)分条列点
        - 正文段首用"&emsp;&emsp;"实现首行缩减，段尾用两个空字符"\\n\\n"实现换行
        - 公式的格式需严格遵守*数学公式格式*
        - 不要使用**给正文加斜体
        
        输出要求*
        请严格按照*markdown格式*输出对应的Markdown格式，不要包含任何其他文本、解释或额外的格式。
        '''
        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': '请严格按照要求生成Markdown格式的文本，不要更改原文本的内容和结构。'},
        ]

        response = self.chat.invoke(prompts).content
        return {"method_part": response}

    def chart_extract_node(self, state: PaperWritingState):
        print("\n>>> 开始进行抽取表格")

        map = {
            0: '一',
            1: '二',
            2: '三',
            3: '四',
            4: '五',
            5: '六',
            6: '七',
            7: '八',
            8: '九',
        }
        # 实验结果
        outcome = state["code_result"]["outcome"]
        outcome_str = []
        for i, oc in enumerate(outcome):
            outcome_str.append(
                f'- 步骤{map[i]}: {oc}'
            )
        outcome_str = '\n'.join(outcome_str)
        prompt = f'''
        你是一个实验结果信息分析专家。

        *任务*
        给定*实验结果*，你需要重中抽取可以表格形式展示的数据，并以Markdown表格的形输出。

        *实验结果*
        {outcome_str}
        
        *抽取格式*
        包含两部分
        - 表格说明: 指出这是第几步、进行什么处理后生成的结果
        - 表格: Markdown表格的形式抽取数据（形式如"| text | text | text |"）

        输出要求*
        JSON是最外层且唯一的输出。
        你必须只返回一个有效的 JSON 对象，不要包含任何其他文本、Markdown 代码块、解释或额外的格式，格式如下:
        {{"charts": [{{"describe": "表格说明", "chart": "Markdown格式的表格"}}]}}
        '''
        prompts = [
            {'role': 'system', 'content': prompt},
        ]
        class Chart(BaseModel):
            describe: str = Field(description="表格说明")
            chart: str = Field(description="Markdown格式的表格")
        class ChartList(BaseModel):
            charts: List[Chart] = Field(description="以List形式存储提取到的全部表格")

        raw_response = self.chat.invoke(prompts)
        raw_content = raw_response.content
        cleaned_content = clean_json_response(raw_content)  # 清理响应
        data = json.loads(cleaned_content)
        response = ChartList(**data)

        ret = []
        for chart in response.charts:
            ret.append(
                {
                    "describe": chart.describe,
                    "chart": chart.chart,
                }
            )

        return {"chart": ret}


    def result_write_node(self, state: PaperWritingState):
        print('\n>>> 实验结果撰写节点')

        map = {
            0: '一',
            1: '二',
            2: '三',
            3: '四',
            4: '五',
            5: '六',
            6: '七',
            7: '八',
            8: '九',
        }
        # 研究方法
        modeling_steps = state["code_result"]["modeling_steps"]
        method_detail = []
        for i, step in enumerate(modeling_steps):
            method_detail.append(
                f'- 步骤{map[i]}: {step}'
            )
        method_detail = '\n'.join(method_detail)
        # 抽取的表格
        charts = []
        for i, chart in enumerate(state["chart"]):
            charts.append(
                f'-{i+1} 描述: {chart["describe"]}\n{chart["chart"]}'
            )
        charts = '\n'.join(charts)
        # 实验结果
        outcome = state["code_result"]["outcome"]
        outcome_str = []
        for i, oc in enumerate(outcome):
            outcome_str.append(
                f'- 步骤{map[i]}: {oc}'
            )
        outcome_str = '\n'.join(outcome_str)
        # 图片路径
        img_dir = os.path.join(str(self.project_dir), 'img')
        img_name = os.listdir(img_dir)
        img_path = '\n'.join([os.path.join(img_dir, name) for name in img_name if '.' in name])
        prompt = f'''
        你是一个论文写作专家，现在你正在撰写论文的实证研究部分。
        *背景说明*
        给定的*研究方法*和*实验结果*，你需要以给定Mardown的格式撰写实验结果，字数为4000-5000字。
        
        *研究主题*
        {state["theme"]}
        
        *研究方法*
        {method_detail}
        
        *实验结果*
        {outcome_str}
        
        *表格*
        以下可使用表格是从*实验结果*中抽取得到。
        {charts}
        
        *撰写格式*
        - ### 作为小标题标志
        - 可拆分为几部分进行论述，用1/2/3...作为小标题序号
        - 不用包含”实证研究“大标题
        - 对于每一小标题的明细内容，可使用无序编号(*)分条列点
        - 正文段首用"&emsp;&emsp;"实现首行缩减，段尾用两个空字符"\\n\\n"实现换行
        - 可适当穿插图片，生成的可使用的图片路径见*图片路径*，图片引用格式见*图片引用格式*
        - 除图片的注释说明外，不要使用**给正文加斜体
        
        *图片引用格式*
        - 插入的图片得单独成段，因此上一段得在段尾使用两个空字符"\\n\\n"换行
        - 图片得有注释
        - 图片引用格式为 "![](图片路径)  "，注释格式为"&emsp;&emsp;*图: 图片注释*\\n\\n"，注意有末尾分别为"  "和"\\n\\n"
        - 示例:
        "![](D:\AiProgram\Thesis\Finance\img\描述性统计_箱线图.jpg)  &emsp;&emsp;*图: 各变量分布的箱线图，验证对数变换的有效性*\\n\\n"
        
        *撰写内容要求*
        - 不能提及文件名，只能提及处理的字段名
        - 除了文字说明外，还需要以表格的形式展示结果。可使用的表格见*表格*，若其数据与*实验结果*存在冲突，以*实验结果*为准
        - 以专业学术的语言论述实验过程和结果，而不是仅仅对步骤进行描述
        
        *图片路径*
        {img_path}
        
        *输出要求*
        请严格按照*撰写格式*输出对应的Markdown格式，不要包含任何其他文本、解释或额外的格式。
        '''

        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': '请严格按照要求生成Markdown格式的方法说明文本。'},
        ]

        response = self.chat.invoke(prompts).content
        return {"result_part": response}

    def introduction_conclusion_write_node(self, state: PaperWritingState):
        print('\n>>> 引言、结论撰写节点')
        prompt = f'''
        你是一个论文写作专家，现在你正在撰写论文的引言部分和结论部分。
        *背景说明*
        给定论文的主题、文献综述和实验结果，写出论文的引言(1500-2000字)、结论和展望(1000-1500字)。
        
        *论文主题*
        {state["theme"]}
        *文献综述*
        {state["literature_review"]}
        *实验结果*
        {state["result_part"]}


        *撰写格式*
        - ### 作为小标题标志
        - 可拆分为几部分进行论述，用1/2/3...作为小标题序号
        - 不用包含“引言”、“结论和展望”大标题
        - 对于每一小标题的明细内容，可使用无序编号(*)分条列点
        - 正文段首用"&emsp;&emsp;"实现首行缩减，段尾用两个空字符"\\n\\n"实现换行
        - 不要使用**给正文加斜体

        *输出要求*
        引言(introduction)、结论和展望(conclusion)都需请严格按照*撰写格式*输出对应的Markdown格式，不要包含任何其他文本、解释或额外的格式。
        最后以JSON格式分别存储引言、结论和展望: {{"introduction": "Markdown格式的引言", "conclusion": "Markdown格式的结论和展望"}}
        '''
        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': '请严格按照*输出要求*的格式生成结果。'},
        ]
        class IC(BaseModel):
            introduction: str = Field(description='Markdown格式的引言')
            conclusion: str = Field(description='Markdown格式的结论和展望')

        structured_model = self.chat.with_structured_output(IC)
        response = structured_model.invoke(prompts)

        return {"introduction_part": response.introduction, "conclusion_part": response.conclusion}

    def reference_node(self, state: PaperWritingState):
        print('\n>>> 开始生成参考文献')
        def review_structure(text: str):
            errors = []
            pattern1 = re.compile(r"\((?P<author>.[^,.!，。！]{2,30}),\s(?P<year>\d{4})\)", re.UNICODE)
            pattern2 = re.compile(r"[，。](?P<author>.[^,.!，。！]{2,10})\((?P<year>\d{4})\)")
            i = 1
            for p in (pattern1, pattern2):
                for match in p.finditer(text):
                    author = match.group("author")
                    year = match.group("year")
                    for meta in state['papers_meta']:
                        if author == meta.author and str(meta.year) == year:
                            errors.append(f'[{i}]{meta.author}, {meta.title}({meta.year}), {meta.public}, {meta.url}')
                            i += 1
                            break
            return '\n'.join(errors)
        return {"reference_part": review_structure(state["literature_review"])}

    def assemble_node(self, state: PaperWritingState):
        print('\n>>> 开始组装论文')

        thesis = '\n'.join(
            [
                f'# {state["theme"]}',
                f'**关键词**: {", ".join(state["search_queries"])}',
                "\n## 一、引言",
                state["introduction_part"],
                "\n## 二、行文大纲",
                state["thing_write_part"],
                '\n## 三、文献综述',
                state["literature_review"],
                '\n## 四、数据指标说明',
                state["indicator_part"],
                '\n## 五、方法说明',
                state["method_part"],
                '\n## 六、实验结果',
                state["result_part"],
                '\n## 七、结论和展望',
                state["conclusion_part"],
                '\n## 参考文献',
                state["reference_part"],
            ]
        )
        return {'thesis': thesis}

    def format_node(self, state: PaperWritingState):
        print("\n>>> 格式节点")

        markdown_content = state["thesis"]
        # ----------------------
        # 1. 从论文提取标题作为文件名
        # ----------------------
        title = state["theme"]

        # ----------------------
        # 2. 保存 Markdown 文件
        # ----------------------
        md_filename = f"{title}.md"
        md_path = os.path.join(self.project_dir, md_filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            print(f'学术MD保存成功！路径为{md_path}')

        # ----------------------
        # 3. Markdown → 导出 PDF
        # ----------------------
        def convert_md_to_pdf(md_file, pdf_file):
            """
            修正后的转换函数
            """
            # 修正的参数
            extra_args = [
                '--pdf-engine=xelatex',
                # '--toc',
                # '--number-sections',
                '--syntax-highlighting=pygments',  # 替代 --highlight-style
                '--variable=subparagraph',
                '--variable=papersize:a4',
                '--variable=geometry:margin=2.5cm',
                # 1. 指定完整字体，解决符号缺失
                '-V', 'mathfont=Latin Modern Math',
                '-V', 'CJKmainfont=Microsoft YaHei',
                # 2. 启用完整的 Unicode 支持
                '--pdf-engine-opt=-shell-escape',
                '--pdf-engine-opt=-8bit',
            ]

            try:
                output = pypandoc.convert_file(
                    md_file,
                    'pdf',
                    outputfile=pdf_file,
                    extra_args=extra_args
                )
                return output
            except Exception as e:
                pass
                # 尝试使用命令行
                cmd = ['pandoc', md_file, '-o', pdf_file] + extra_args
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    pass

        # 导出 PDF
        pdf_path = os.path.join(self.project_dir, f"{title}.pdf")
        convert_md_to_pdf(md_path, pdf_path)

        print(f"md_path: {md_path} | pdf_path: {pdf_path} | title: {title}")

        return {}

    def run_graph(
            self,
            user_input: str,
            thread_id: str,
            max_review_iteration: int = 2,
    ):
        """运行图并处理用户交互"""
        initial_state = {
            "user_input": user_input,
            "max_review_iteration": max_review_iteration
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
            continue

        # 返回最终代码执行结果
        state = self.graph.get_state(config).values
        return state["thesis"]

if __name__ == '__main__':
    agent = SupAgent()
    result = agent.run_graph(
        user_input =  "我想写一篇有关人口对GDP影响因数的论文。",
        thread_id = '29819'
    )
    print(result)


