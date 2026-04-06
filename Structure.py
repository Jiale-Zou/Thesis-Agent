import os
import re
import warnings
import requests
import json
import uuid
import asyncio
import markdown
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any, Annotated


from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langchain_core.messages import AnyMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from ThesisAgent.BlackBox import CodeGenerationAgent
from Model import chat_model, parser

warnings.filterwarnings('ignore')

''' 论文写作Agent
工作流：规划 → 检索 → 解析 → 开题报告 → 撰写 → 评审
'''

### 任务配置 ###
LLM = chat_model('qwen', temperature=0.7) # 模型
PATH = 'D:\\AiProgram\\Thesis\\' # 工作目录
project_name = 'Finance' # 项目名称
project_dir = os.path.join(PATH, project_name)
invalid_chars = r'[\\/:*?"<>|]' # 不计入字数统计的符号


class PaperWritingState(TypedDict):
    '''
    定义全局状态State的属性
    '''
    user_input: str

    # 规划阶段
    search_queries: List[str]  # 生成的检索关键词

    # 检索阶段
    # raw_papers_metadata: List[Dict]  # 原始论文元数据（标题、DOI、链接）
    # raw_papers_localpath: List[str] # 论文本地保存路径

    # 检索阶段
    parsed_papers: List[Dict]  # 解析后的论文内容（作者、年份、摘要、doi）

    # 开题报告阶段
    detailed_outline: str # 明细版结构化大纲
    literature_review: str  # 文献综述摘要
    methodology: Dict # 方法说明

    # 大纲和文献综述检查
    outline_check: str
    literature_check: str
    check_state: bool # 是否通过

    # 代码执行结果
    code_result: str

    # 格式检查
    len_check: str
    structure_check: str

    # 写作与评审
    draft: str  # 初稿
    review_state: bool # 是否通过

    current_step: str  # 当前节点（用于调试）

def plan_node(state: PaperWritingState):
    print(">>> 规划节点")
    writer = get_stream_writer()
    writer({'node': 'plan_node'})

    prompt = '''
    你是一个论文写作专家。
    
    *任务*
    根据用户的要求，生成该论文有关的'检索关键词'，要求全面涵盖主题，最多为4个。
    
    *分析示例*
    **用户输入**: "我要写一篇有关ESG与股票风险关系的论文，具体来说，需要先采用某种风险测度衡量股票风险，然后选取某些指标来衡量股票ESG水平，最后再实证ESG水平与股票风险测度的关系。"
    **分析过程**:
        1. **明确主题**: 用户的主题为'ESG与股票风险'，其中涉及到两个关键字'ESG'和'股票分析'。
        2. **分析用户具体要求**: 用户提出了具体的要求，即分为三方面考虑，一是定义'风险测度'来衡量股票风险，二是选取指标衡量'ESG水平'，三是研究'相关关系'。
        3. **确定文章检索的关键字**: 根据对用户意图的分析，找到有关的的关键字为'ESG评分'、'ESG指标'、'股票风险测度'、'VaR'、'ES'、'相关性研究'、'因果关系'等。
    **输出**: 严格遵循JSON的格式，不要有任何其他内容和其他形式的输出，输出的格式如下。
        {"keyword": ["","",""]}
    '''
    prompts = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': state['user_input']},
    ]
    response = LLM.invoke(prompts).content
    json_response = {'keyword': []}
    try:
        j = json.loads(response)
        for key, value in j.items():
            json_response[key.lower()] = value
    except Exception as e:
        raise json.JSONDecodeError(f'plan node "JSON loads error": {e}:')

    writer({'plan_step': f"检索关键词为{json_response['keyword'][:4]}"})

    return {'search_queries': json_response['keyword'][:4]}


def search_node(state: PaperWritingState):
    print(">>> 搜索节点")
    writer = get_stream_writer()
    writer({'node': 'sparse_node'})

    mcp_services = json.load(open("mcp_config.json", 'r'))['mcpServers']
    client = MultiServerMCPClient(
        {
            "cnki": mcp_services["cnki"]
        }
    )
    tools = asyncio.run(client.get_tools())
    agent = create_react_agent(
        model=LLM,
        tools=tools,
    )

    # 论文摘要
    prompt = '''
你是一个论文总结分析专家。

*任务*
根据传入检索关键词，调用论文搜索工具搜索相关论文，并提取相关信息，解析论文的作者(author)、年份(year)、摘要(abstract)、doi。

*字数限制*
摘要: 200-300字。

*输出要求*
严格遵循JSON的格式。不要有任何其他内容和其他形式的输出，输出的格式如下。
[{"author": "", "year": "", "abstract": "", "doi":""}]
'''
    prompts = {
        "messages": [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f"检索的关键字为: {state['search_queries']}。请严格按照要求检索信息。"}
        ]
    }
    response = asyncio.run(agent.ainvoke(prompts))["messages"][-1].content
    json_response = []
    try:
        json_response = json.loads(response)
    except Exception as e:
        raise json.JSONDecodeError(f'plan node "JSON loads error": {e}:')

    if not json_response: # search mcp可能有问题，这段代码仅为了测试
        json_response = [
  {
    "author": "Bloom, David E.; Canning, David; Sevilla, Jaypee",
    "year": "2003",
    "abstract": "This paper examines the relationship between demographic change and economic growth, focusing on the 'demographic dividend'—the boost to per capita GDP from a rising working-age population share relative to dependents. It uses cross-country panel data to show that favorable age structure significantly accelerates growth via increased labor supply, savings, and human capital investment, and identifies policy conditions for capturing this dividend.",
    "doi": "10.1086/344171"
  },
  {
    "author": "Mason, Andrew; Lee, Ronald",
    "year": "2011",
    "abstract": "The study analyzes how population aging affects GDP and economic sustainability by redefining demographic dependency through the 'support ratio' (working-age population relative to effective consumers). It demonstrates that aging reduces the demographic dividend, lowers potential GDP growth, and increases fiscal pressure on pension and healthcare systems, while highlighting human capital and productivity gains as key mitigators.",
    "doi": "10.1111/j.1749-6632.2011.06050.x"
  },
  {
    "author": "Jones, Charles I.",
    "year": "2019",
    "abstract": "This paper revisits the long-run link between population size, human capital, and GDP growth in a semi-endogenous growth framework. It argues that larger populations drive innovation and knowledge spillovers, boosting total factor productivity and long-run GDP levels, while slower population growth reduces steady-state growth rates, with implications for developed economies facing declining fertility.",
    "doi": "10.1257/jep.33.2.81"
  }
]

    writer({'search_step': f"共检索到{len(json_response)}篇论文。"})
    return {'parsed_papers': json_response}

def report_node(state: PaperWritingState):
    print(">>> 报告节点")
    writer = get_stream_writer()
    writer({'node': 'report_node'})

    if state.get('check_state', True):
        prompt = '''
你是一个论文写作专家。

*任务*
给定'用户要求', "参考文献"，请你按照学术规范生成生成论文的开题报告。包含如下方面：
1. 确定论文的文章标题和主题
2. 根据主题和参考文献，生成学术的文献综述。（要求800-1000字）。
3. 对于研究方法，给出需要提供的数据字段、字段含义、具体的研究方法、模型以及如何用该方法模型研究该数据
4. 生成的结构化大纲必须分为"引言", "文献综述", "研究方法", "指标定义"(可以不包含), "研究结果", "总结与展望"，且详细介绍该部分具体的工作

*结构化大纲示例（需要细化每一部分的具体工作）*
文章可分为六部分展开。\
一、引言：对研究的背景与动机进行介绍。\
二、文献综述：需要根据已有的研究进行总结，分析以往的研究成果、优点以及不足。\
三、研究方法：结合以往学者采取的研究方法，并考虑到方法的创新性，为该主题选取合适的研究方法。\
四、指标定义：包括'股票风险测度'的定义，对'ESG'水平衡量的定义，以及其他研究中涉及到的变量的定义。
五、研究结果：先对使用的数据进行来源、描述性统计说明，然后使用研究方法对数据进行研究，得出响应的实证结果。
六、总结与展望：总结研究的成果，并对研究中的优缺点进行分析，最后对未来的研究方向进行展望。

*要求*
1. 结构化大纲(detailed_outline)的具体指导要紧扣主题，包含"引言", "文献综述", "研究方法", "指标定义"(可以不包含), "研究结果", "总结与展望"这几个章节。
2. 文献综述(literature_review)的格式要符合学术规范，只能引用给定的参考文献，且对于引用的部分，要在后面以"(author, year)"的格式标注对应的参考文献。
3. 研究方法(methodology)以JSON格式表示，包含'data','method'两部分。其中'data'对应的值为列表(可以为空，若为空，则表示不需要进行编程)，列表中的每个元素表示数据名称、数据类型、描述，\
以{"name": "", "type": "", "describe": ""}的JSON格式储存；'method'对应的值为字符串，是对具体的研究方法、模型以及如何用该方法模型研究该数据的描述。

**输出**: 严格遵循JSON的格式，不要有任何其他内容和其他形式的输出，输出的格式如下。
    {"detailed_outline": "", "literature_review": "", "methodology": {"data": [{"name":"", "type":"", "describe":""}], "method": ""}}
'''
        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'用户要求: "{state["user_input"]}"。参考文献: "{state["parsed_papers"]}"'},
        ]
        response = LLM.invoke(prompts).content
        json_response = {"detailed_outline": "", "literature_review": "", "methodology": {"data": [{"name":"", "type":"", "describe":""}], "method": ""}}
        try:
            j = json.loads(response)
            for key, value in j.items():
                json_response[key.lower()] = value
        except Exception as e:
            raise json.JSONDecodeError(f'plan node "JSON loads error": {e}:')
    else:
        prompt = f'''
你是一个论文写作专家，现在你需要根据*反馈*修正*论文结构化大纲和文献综述*。

*要求*
1. 结构化大纲(detailed_outline)的具体指导要紧扣主题，包含"引言", "文献综述", "研究方法", "指标定义"(可以不包含), "研究结果", "总结与展望"这几个章节。
2. 文献综述(literature_review)的格式要符合学术规范(要求800-1000字)，只能引用给定的参考文献，且对于引用的部分，要在后面以"(author, year)"的格式标注对应的参考文献。

*初稿*
- 结构化大纲:
{state["detailed_outline"]}
- 文献综述:
{state["literature_review"]}

*反馈*
- 结构化大纲:
{state["outline_check"]}
- 文献综述:
{state["literature_check"]}

**输出**: 严格遵循JSON的格式，不要有任何其他内容和其他形式的输出，输出的格式如下。
    {{"detailed_outline": "", "literature_review": ""}}
'''
        prompts = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'用户的要求为: "{state["user_input"]}"。请严格按照反馈修正结构化大纲和文献综述'},
        ]
        response = LLM.invoke(prompts).content
        json_response = {"detailed_outline": "", "literature_review": ""}
        try:
            j = json.loads(response)
            for key, value in j.items():
                json_response[key.lower()] = value
        except Exception as e:
            raise json.JSONDecodeError(f'plan node "JSON loads error": {e}:')

    writer({'report_step': f"文献综述:\n{json_response['literature_review'][:100]}"})

    return {
        "detailed_outline": json_response['detailed_outline'],
        "literature_review": json_response['literature_review'],
        "methodology": json_response['methodology'] if 'methodology' in json_response else state['methodology'],
    }

def structure_check_node(state:PaperWritingState):
    print(">>> 结构检查节点")
    writer = get_stream_writer()
    writer({'node': 'structure_check_node'})

    def outline_check(text: str):
        errors = []
        require_chapter = ["引言", "文献综述", "研究方法", "研究结果", "总结与展望"]
        for x in require_chapter:
            if x not in text:
                errors.append(f'缺少"{x}"章节')
        return ' | '.join(errors)

    def word_check(text: str):
        length = len(text)
        if 800 <= length <= 1000:
            return ''
        return f'文献综述要求文字为800-1000，实际为{length}字'

    def review_check(text: str):
        errors = []
        words = word_check(text)
        if words:
            errors.append(words)
        author_year_pairs = {}
        for x in state['parsed_papers']:
            author_year_pairs[x['author']] = x['year']
        pattern = re.compile(r"\((?P<author>.+?)[,，]\s*(?P<year>\d{4})\)", re.UNICODE)
        for match in pattern.finditer(text):
            author = match.group("author")
            year = match.group("year")
            if (author in author_year_pairs) and author_year_pairs[author] == year:
                continue
            else:
                errors.append(f'引用了author="{author}",year="{year}"，与参考文献不符。')
        return ' | '.join(errors)

    outline = outline_check(state['detailed_outline'])
    review = review_check(state['literature_review'])

    writer({'structure_check_step': f"大纲check: {outline}\n文献综述check: {review}"})

    if (not outline) and (not review):
        check_state = True
    else:
        check_state = False

    return {"outline_check": outline, "literature_check": review, "check_state": check_state}

def code_node(state: PaperWritingState):
    print(">>> 代码节点")
    writer = get_stream_writer()
    writer({'node': 'code_node'})

    methodology = state['methodology']
    methodology_data = methodology['data']
    methodology_method = methodology['method']

    if methodology_data:
        agent = CodeGenerationAgent(LLM)
        result = agent.run_with_user_interaction(
            uuid.uuid4().hex,
            project_name,
            methodology_data,
            methodology_method,
        )

        return {"code_result": result}
    return

def write_node(state: PaperWritingState):
    print(">>> 撰写节点")
    writer = get_stream_writer()
    writer({'node': 'write_node'})

    if not state.get('review_state', False):
        # 找到write_code步骤输出的全部图片(路径名为图片的含义)
        img_path = os.path.join(project_dir , 'img')
        img_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
        image_files = []
        for file in os.listdir(img_path):
            if file.lower().endswith(img_extensions): # 转小写判断，避免大小写问题
                desc, end = file.rsplit('.', 1)
                desc = desc.rsplit('\\', 1)[0]
                image_files.append(f"Img_path: '{file}' ; Img_describe: {desc}")
        image_files = '\n'.join(image_files)

        data = "\n".join([
                f"- {f['name']}: {f['type']} ({f['describe']})"
                for f in state["methodology"]["data"]
            ])

        prompt = f'''
你是一个论文写作专家。

*任务*
给定"论文结构化大纲", "文献综述", "方法论", "数据说明", "运行结果", "运行结果图片路径与说明"，请你按照学术规范写一篇论文。

*要求*
1. 严格按照"论文结构化大纲"组织行文思路和结构
2. "文献综述"按照传入的照抄即可
3. 对于模型方法和数据说明部分，可适当插入公式
4. 如果给定了图片，可在适当位置插入图片
5. 论文字数为15000-20000字

*论文结构化大纲*
{state["detailed_outline"]}

*文献综述*
{state["literature_review"]}

*方法论*
{state["methodology"]['method']}

*数据说明*
{data}

*运行结果*
{state.get("code_result", "")}

*运行结果图片路径与说明*
{image_files}

*Markdown格式规范*
绝对不允许出现任何自然语言解释、前言、总结、对话、注释、多余说明。
不允许使用 ```markdown 代码块包裹，不允许加任何多余文字。
格式规则：
- 标题仅使用 #、##、###，层级严格对应。
- 段落正常书写，不使用粗体以外的强调。
- 列表使用 - 或 1. 格式。
- 公式使用 $...$ 或 $$...$$。
- 图表使用 ![](url) 或 | 表格 | 语法

**输出**: 你必须严格、唯一、仅按照*Markdown格式规范*输出完整论文。
请在下面标签中输出Markdown格式论文。
<markdown>
 # markdown thesis
</markdown>
'''
    else:
        prompt = f'''
请你修正论文的markdown要求。

*任务*
不改变论文的内容(包括公式、文献综述、图片链接、论文中提及的数据等等)，只根据*评审建议*的字数要求和格式要求调整论文Markdown格式。

*论文初稿*
{state['draft']}

*评审建议*
字数: {state['len_check']}
格式: {state['structure_check']}

*Markdown格式规范*
绝对不允许出现任何自然语言解释、前言、总结、对话、注释、多余说明。
不允许使用 ```markdown 代码块包裹，不允许加任何多余文字。
格式规则：
- 标题仅使用 #、##、###，层级严格对应。
- 段落正常书写，不使用粗体以外的强调。
- 列表使用 - 或 1. 格式。
- 公式使用 $...$ 或 $$...$$。
- 图表使用 ![](url) 或 | 表格 | 语法

**输出**: 你必须严格、唯一、仅按照*Markdown格式规范*输出完整论文。
请在下面标签中输出修正的Markdown格式论文。
<markdown>
 # markdown thesis
</markdown>
'''
    prompts = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': f'按照要求在指定区域生成论文。'},
    ]

    response = LLM.invoke(prompts).content
    # 提取 markdown论文
    markdown_pattern = r"<markdown>(.*?)</markdown>"
    markdown_match = re.search(markdown_pattern, response, re.DOTALL | re.MULTILINE)
    markdown = markdown_match.group(1).strip() if markdown_match else ""

    return {"draft": markdown}


def review_node(state: PaperWritingState):
    print(">>> 评审节点")
    writer = get_stream_writer()
    writer({'node': 'review_node'})

    def count_effective_words(text: str) -> int:
        """计算学术有效字数：剔除标题、公式、代码、链接、符号，只算正文"""
        # 1. 移除标题 (# ...)
        text = re.sub(r'^#+\s*.+', '', text, flags=re.MULTILINE)
        # 2. 移除行内公式 $...$
        text = re.sub(r'\$[^$]+\$', '', text)
        # 3. 移除块公式 $$...$$
        text = re.sub(r'\$\$[\s\S]*?\$\$', '', text)
        # 4. 移除链接 [text](sslocal://flow/file_open?url=url&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=) → 保留 text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # 5. 移除图片![…]
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # 6. 移除代码块 ```...```
        text = re.sub(r'```[\s\S]*?```', '', text)
        # 7. 移除粗体/斜体
        text = re.sub(r'\*\*?|\__?', '', text)
        # 8. 分割成单词（中文按字，英文按词）
        words = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]', text)
        return len(words)

    def check_markdown_structure(text: str) -> list:
        """检查论文格式：标题层级、必须章节、违规内容"""
        errors = []
        lines = [line.strip() for line in text.split('\n')]

        # 必须章节
        required_sections = {
            "#",
            "## 1", "## 2", "## 3", "## 4", "## 5",
            "## 引言", "## 文献综述", "## 研究方法", "## 研究结果", "## 总结与展望"
        }
        found = set()

        prev_level = 0  # 上一级标题 # 数量

        for i, line in enumerate(lines, 1):
            # 检查标题
            if line.startswith('#'):
                match = re.match(r'(#+)', line)
                if match:
                    level = len(match.group(1))
                    # 标题层级跳跃（例如 # -> ###）
                    if level > prev_level + 1:
                        errors.append(f"第{i}行：标题层级跳跃 {prev_level} → {level}")
                    prev_level = level

                    # 记录章节
                    if level == 1:
                        found.add("#")
                    elif level == 2:
                        found.add(line)

        # 必须章节缺失
        missing = [s for s in required_sections if any(s in sec for sec in found) is False]
        if missing:
            errors.append(f"缺少章节：{' | '.join(missing[:5])}...")

        # 检查违规前缀废话
        forbidden_prefix = [
            "好的", "以下是", "我将为您", "请查收", "下面给",
            "论文如下", "开始输出", "markdown", "```"
        ]
        first_lines = '\n'.join(lines[:10]).lower()
        for w in forbidden_prefix:
            if w in first_lines:
                errors.append(f"包含违规前缀：{w}（不符合纯论文要求）")
                break

        return errors

    markdown_paper = state['draft']

    words = count_effective_words(markdown_paper)
    errors = check_markdown_structure(markdown_paper)
    if 15000 <= words <= 20000 and len(errors) == 0:
        review_state = True
    else:
        review_state = False

    return {
        'len_check': f'论文有效字数为{words}, 要求的字数为15000-20000',
        'structure_check': ' | '.join(errors),
        'review_state': review_state,
    }

def format_node(state: PaperWritingState):
    print(">>> 格式节点")
    writer = get_stream_writer()
    writer({'node': 'format_node'})

    from weasyprint import HTML

    markdown_content = state["draft"]
    os.makedirs(project_dir, exist_ok=True)
    # ----------------------
    # 1. 从论文提取标题作为文件名
    # ----------------------
    title_match = re.search(r"^#\s+(.+)", markdown_content, re.MULTILINE)
    if title_match:
        title = re.sub(r'[\\/*?:"<>|]', "", title_match.group(1))
    else:
        title = f"论文_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ----------------------
    # 2. 保存 Markdown 文件
    # ----------------------
    md_filename = f"{title}.md"
    md_path = os.path.join(project_dir, md_filename)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # ----------------------
    # 3. 转换为 HTML → 导出 PDF
    # ----------------------
    html_body = markdown.markdown(
        markdown_content,
        extensions=[
            "tables",  # 支持表格
            "fenced_code",  # 支持代码块
            "toc",  # 支持目录
            "footnotes",
            "attr_list",
            "md_in_html",
            "pymdownx.arithmatex",  # 公式支持
            "pymdownx.magiclink",
            "pymdownx.betterem",
        ],
        extension_configs={
            "pymdownx.arithmatex": {
                "generic": True
            }
        }
    )
    html_body = re.sub(r"\$\$(.*?)\$\$", r'<span class="math display">\\[\1\\]</span>', html_body, flags=re.DOTALL)
    html_body = re.sub(r"\$(.*?)\$", r'<span class="math inline">\\(\1\\)</span>', html_body)

    # 美化学术 PDF 样式
    full_html = f'''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{
                    font-family: "Microsoft YaHei", SimSun, Arial, sans-serif;
                    font-size: 14.5px;
                    line-height: 1.7;
                    margin: 2cm;
                    color: #222;
                }}
                h1, h2, h3, h4, h5 {{
                    color: #000;
                    page-break-after: avoid;
                    page-break-inside: avoid;
                }}
                p {{
                    text-align: justify;
                    margin: 0.7em 0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                    page-break-inside: avoid;
                }}
                th, td {{
                    border: 1px solid #555;
                    padding: 0.6em;
                    text-align: center;
                }}
                img {{
                    max-width: 100% !important;
                    height: auto;
                    display: block;
                    margin: 1em auto;
                    page-break-inside: avoid;
                }}
                .math.display {{
                    text-align: center;
                    font-size: 1.1em;
                    margin: 1em 0;
                }}
                pre code {{
                    background: #f3f3f3;
                    padding: 1em;
                    display: block;
                    border-radius: 5px;
                    page-break-inside: avoid;
                }}
            </style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        '''

    # 导出 PDF
    pdf_path = os.path.join(project_dir, f"{title}.pdf")
    HTML(string=full_html).write_pdf(
        pdf_path,
    )

    writer({'format_step': f"md_path: {md_path} | pdf_path: {pdf_path} | title: {title}"})

    return {}

# 添加节点
builder = (StateGraph(PaperWritingState)
    .add_node('plan_node', plan_node)
    .add_node('search_node', search_node)
    .add_node('report_node', report_node)
    .add_node('structure_check_node', structure_check_node)
    .add_node('code_node', code_node)
    .add_node('write_node', write_node)
    .add_node('review_node', review_node)
    .add_node('format_node', format_node))
# 定义边函数
def review_route_func(state: PaperWritingState):
    '''若评审通过，则进入格式化节点'''
    if state['review_state']:
        return 'format_node'
    else:
        return 'write_node'
def structure_route_func(state: PaperWritingState):
    '''若需要外部传入data(编程)，则进入撰写节点'''
    if not state['check_state']:
        return 'report_node'
    if state['methodology']['data']:
        return 'code_node'
    else:
        return 'write_node'
# 添加边
builder.add_edge(START, 'plan_node')
builder.add_edge('plan_node', 'search_node')
builder.add_edge('search_node', 'report_node')
builder.add_edge('report_node', 'structure_check_node')
builder.add_conditional_edges('structure_check_node', structure_route_func, ['report_node', 'code_node', 'write_node'])
builder.add_edge('code_node', 'write_node')
builder.add_edge('write_node', 'review_node')
builder.add_conditional_edges('review_node', review_route_func, ['format_node', 'write_node'])
builder.add_edge('format_node', END)

# 构建图
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

if __name__ == '__main__':
    config = {
        'configurable': {
            'thread_id': uuid.uuid4(),
        }
    }
    for trunk in graph.stream(
            {'user_input': '我想写一篇人口对GDP影响的论文。'},
            config):
        print(trunk)

