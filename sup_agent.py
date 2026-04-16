import os
import re
import subprocess
import warnings
from pathlib import Path

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
from retry_utils import retry_call, retry_json_parse, RetryPolicy
from cursor_skill_loader import build_keyword_messages

warnings.filterwarnings('ignore')

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except Exception:
    SqliteSaver = None

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
        self.checkpointer = self._build_checkpointer(config)
        self.retry_policy = RetryPolicy(max_attempts=3) # 最大三次重试
        self.graph = self._build_graph()

    def _build_checkpointer(self, config: Dict[str, Any]):
        """
        Time-Travel 需要持久化检查点；优先 SqliteSaver，失败时回退 MemorySaver。
        """
        db_path = config.get("checkpoint_db", "")
        if not db_path:
            db_path = str((Path(self.project_dir) / "logs" / "agent.sqlite").resolve())
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        if SqliteSaver is not None:
            try:
                saver = SqliteSaver.from_conn_string(db_path)
                return saver
            except Exception as e:
                print(f"[warn] SqliteSaver初始化失败，回退到MemorySaver: {e}")
        else:
            print("[warn] 当前环境无SqliteSaver，回退到MemorySaver。")
        return MemorySaver()

    def _mk_config(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        '''构造config'''
        configurable = {"thread_id": thread_id}
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable}

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
        workflow.add_edge('method_write_node', 'chart_extract_node')
        workflow.add_edge('chart_extract_node', 'result_write_node')
        workflow.add_edge('result_write_node', 'introduction_conclusion_write_node')
        workflow.add_edge('introduction_conclusion_write_node', 'reference_node')
        workflow.add_edge('reference_node', 'assemble_node')
        workflow.add_edge('assemble_node', 'format_node')
        # 编译图
        return workflow.compile(checkpointer=self.checkpointer.__enter__()) # 为保证SqliteSaver生命周期与graph一压样，前面返回连接，这里传入实例.__enter__()


    def _literature_check_route(self, state: PaperWritingState):
        '''边函数'''
        return state["next_node"]

    def keyword_node(self, state: PaperWritingState):
        print(">>> 关键字节点")

        prompts = build_keyword_messages(
            'paper-keyword',
            user_input=f'## 用户要求\n{state["user_input"]}'
        )

        class KeyWord(BaseModel):
            keyword: List[str] = Field(description='检索关键词')

        structured_model = self.chat.with_structured_output(KeyWord)
        response = retry_call(lambda: structured_model.invoke(prompts), policy=self.retry_policy)

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
            resp = retry_call(
                lambda: asyncio.run(
                    search_google_scholar_key_words.ainvoke(
                        {
                            'query': query,
                            'num_results': 10
                        }
                    )
                ),
                policy=self.retry_policy,
            )
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
            ## 文献综述初稿
            {state['literature_review']}
            ## 初稿不合规
            {state['literature_review_check']}
            '''

        prompts = build_keyword_messages(
            'literature-review',
            render={'task': '写出文献综述' if state['next_node']=='literature' else '参照*初稿不合规*，修改*参考文献*'},
            user_input=f'## 参考文献\n{str_papers_meta}\n## 用户写作要求\n{state["user_input"]}\n{supply_prompt}'
        )

        class Literature(BaseModel):
            literature: str = Field(description='符合格式要求，且字数为3000-4000之间的文献综述')

        structured_model = self.chat.with_structured_output(Literature)
        response = retry_call(lambda: structured_model.invoke(prompts), policy=self.retry_policy)
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

        prompts = build_keyword_messages(
            'plan-thesis',

        )

        prompts = build_keyword_messages(
            'plan-thesis',
            user_input=f'## 用户要求\n"{state["user_input"]}"\n\n## 文献综述\n{state["literature_review"]}'
        )

        class Data(BaseModel):
            name: str = Field('字段名称，用英文表示')
            type: str = Field('字段类型，如int,str,float等')
            description: str = Field('字段描述，对字段的含义进行说明')

        class Structure(BaseModel):
            theme: str = Field('论文题目')
            data: List[Data] = Field('研究数据')
            method: str = Field('研究方法')

        structured_model = self.chat.with_structured_output(Structure)
        response = retry_call(lambda: structured_model.invoke(prompts), policy=self.retry_policy)

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

        prompts = build_keyword_messages(
            'plan-thesis-check',
            user_input=f'## 研究方法\n"{state["method"]}"\n\n## 数据字段\n{data_str}。请按要求进行判断'
        )

        class Data(BaseModel):
            name: str = Field('字段名称，用英文表示')
            type: str = Field('字段类型，如int,str,float等')
            description: str = Field('字段描述，对字段的含义进行说明')

        class Structure(BaseModel):
            valid: Literal["Yes", "No"] = Field('是否合规，只能为Yes或No')
            data: List[Data] = Field('研究数据')
            method: str = Field('研究方法')

        structured_model = self.chat.with_structured_output(Structure)
        response = retry_call(lambda: structured_model.invoke(prompts), policy=self.retry_policy)

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
                {'name': d['name'], 'type': d['type'], 'describe': d['description']}
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
        prompts = build_keyword_messages(
            'study-step',
            user_input=f'## 研究方法\n{method_detail}'
        )
        def _make_text() -> str:
            return retry_call(lambda: self.chat.invoke(prompts).content, policy=self.retry_policy)
        data = retry_json_parse(_make_text, policy=self.retry_policy, cleaner=clean_json_response)
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

        # method = state['method']
        data = state["data"]
        data_str = []
        for d in data:
            data_str.append(
                f'- 字段名称:{d.name}, 字段含义:{d.description}。'
            )
        data_str = '\n'.join(data_str)
        init_files = state["code_result"]["init_files"]

        prompts = build_keyword_messages(
            'indicator-write',
            user_input=f'## 数据文件路径\n{init_files}\n\n## 数据说明\n{data_str}'
        )

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

        prompts = build_keyword_messages(
            'method-write',
            user_input=f'## 研究方法\n{method}\n\n## 研究方法详细步骤\n{method_detail}'
        )

        response = retry_call(lambda: self.chat.invoke(prompts).content, policy=self.retry_policy)
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
        prompts = build_keyword_messages(
            'chart-extract',
            user_input=f'## 实验结果\n{outcome_str}'
        )
        class Chart(BaseModel):
            describe: str = Field(description="表格说明")
            chart: str = Field(description="Markdown格式的表格")
        class ChartList(BaseModel):
            charts: List[Chart] = Field(description="以List形式存储提取到的全部表格")

        def _make_text() -> str:
            return retry_call(lambda: self.chat.invoke(prompts).content, policy=self.retry_policy)
        data = retry_json_parse(_make_text, policy=self.retry_policy, cleaner=clean_json_response)
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

        prompts = build_keyword_messages(
            'result-write',
            user_input=f'## 研究主题\n{state["theme"]}\n\n## 研究方法\n{method_detail}\n\n## 实验结果\n{outcome_str}\n\n## 表格\n以下表格从实验结果中抽取得到。\n{charts}\n\n## 图片路径\n{img_path}'
        )
        response = retry_call(lambda: self.chat.invoke(prompts).content, policy=self.retry_policy)
        return {"result_part": response}

    def introduction_conclusion_write_node(self, state: PaperWritingState):
        print('\n>>> 引言、结论撰写节点')

        prompts = build_keyword_messages(
            'intro-conc-write',
            user_input=f'## 论文主题\n{state["theme"]}\n\n## 实验结果\n{state["result_part"]}'
        )
        class IC(BaseModel):
            introduction: str = Field(description='Markdown格式的引言')
            conclusion: str = Field(description='Markdown格式的结论和展望')

        structured_model = self.chat.with_structured_output(IC)
        response = retry_call(lambda: structured_model.invoke(prompts), policy=self.retry_policy)

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

    def get_current_state(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """读取当前/指定checkpoint的状态快照。"""
        config = self._mk_config(thread_id, checkpoint_id)
        state = self.graph.get_state(config)
        return state.values

    def list_state_history(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        列出状态历史（用于选择checkpoint进行Time-Travel）。
        """
        config = self._mk_config(thread_id)
        if not hasattr(self.graph, "get_state_history"):
            raise RuntimeError("当前langgraph版本不支持 get_state_history。")

        rows: List[Dict[str, Any]] = []
        for i, item in enumerate(self.graph.get_state_history(config)):
            if i >= limit:
                break
            rows.append(
                {
                    "index": i,
                    "checkpoint_id": (item.config or {}).get("configurable", {}).get("checkpoint_id", ""),
                    "next": list(item.next) if getattr(item, "next", None) else [],
                    "values": item.values,
                }
            )
        return rows

    def patch_state(
        self,
        thread_id: str, # 线程ID
        state_patch: Dict[str, Any], # 你要改的内容，Dict
        checkpoint_id: Optional[str] = None, # 要修改哪个历史断点（不填=最新断点）
        as_node: Optional[str] = None, # 内部用，永远不用传
    ) -> Dict[str, Any]:
        """
        人工改 state：
        - 可以在当前线程最新checkpoint改
        - 也可以指定checkpoint_id后改
        """
        config = self._mk_config(thread_id, checkpoint_id)
        kwargs = {}
        if as_node:
            kwargs["as_node"] = as_node
        self.graph.update_state(config, state_patch, **kwargs)
        return self.get_current_state(thread_id, checkpoint_id)

    def resume_from_checkpoint(
        self,
        thread_id: str, # 线程ID
        checkpoint_id: Optional[str] = None, # 可选: 从哪个断点继续（不填=最新）
        state_patch: Optional[Dict[str, Any]] = None, # 可选: 恢复前先修改状态，再继续运行
    ) -> Dict[str, Any]:
        """
        从断点恢复执行：
        1) 可选切换到历史checkpoint
        2) 可选先patch state
        3) 从该状态继续stream
        """
        config = self._mk_config(thread_id, checkpoint_id)
        if state_patch:
            self.graph.update_state(config, state_patch)

        for trunk in self.graph.stream(None, config):
            continue
        return self.graph.get_state(config).values

    def run_graph(
            self,
            user_input: Optional[str], # 你的需求（resume=True 时可不传）
            thread_id: str, # 线程ID
            max_review_iteration: int = 2, # 文献综述最大迭代次数
            resume: bool = False, # True=从断点继续，False=从头跑
            state_patch: Optional[Dict[str, Any]] = None, # 运行前先修改状态
            checkpoint_id: Optional[str] = None,
    ):
        """运行图（支持断点重启 + 人工state修订）"""
        config = self._mk_config(thread_id, checkpoint_id)

        if state_patch:
            self.graph.update_state(config, state_patch)

        if resume:
            stream_input = None
        else:
            if not user_input:
                raise ValueError("resume=False 时 user_input 不能为空。")
            stream_input = {
                "user_input": user_input,
                "max_review_iteration": max_review_iteration
            }

        for trunk in self.graph.stream(
                stream_input,
                config,
        ):
            continue

        state = self.graph.get_state(config).values
        return state.get("thesis", "")

if __name__ == '__main__':
    agent = SupAgent()
    result = agent.run_graph(
        user_input =  None,#"我想写一篇有关人口对GDP影响因数的论文。",
        thread_id = '29819',
        resume = True,
    )
    print(result)


