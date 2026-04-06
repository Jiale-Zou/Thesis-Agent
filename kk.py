# def search_node(state: PaperWritingState):
#     print(">>> 检索节点")
#     writer = get_stream_writer()
#     writer({'node': 'search_node'})
#
#     search_queries = state['search_queries']
#
#     # MCP archiv工具
#     client = MultiServerMCPClient(
#         {
#             "arxiv-paper-mcp": {
#                 "command": "npx",
#                 "args": ["-y", "@langgpt/arxiv-paper-mcp@latest"],
#                 'transport': 'stdio'
#             }
#         }
#     )
#     tools = asyncio.run(client.get_tools())
#     search_tool = [t for t in tools if t.name == "search_arxiv"][0]
#     download_tool = [t for t in tools if t.name == "get_arxiv_pdf_url"][0]
#     # re提取论文信息
#     metadata_pattern = re.compile(
#         r'\d+\.\s*\*\*(.*?)\*\*\s*\n\s*ID:\s*(.*?)\s*\n\s*发布日期.*?\n\s*作者:\s*(.*?)\s*\n\s*摘要:\s*(.*?)(?=\s*\n\s*(?:URL:|$))',
#         re.DOTALL)
#     url_pattern = re.compile(r'https?://[^\s]+')
#     # 下载论文pdf到本地
#     def download(url, localpath):
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         }
#         response = requests.get(url, headers=headers, timeout=30)
#         response.raise_for_status()  # 检查HTTP错误
#         with open(localpath, 'wb') as f:
#             f.write(response.content)
#
#     paper_metadata = []
#     paper_localpath = []
#     for query in search_queries:
#         text = asyncio.run(
#             search_tool.ainvoke({
#                 "query": query,
#                 "maxResults": 5
#             })
#         )[0]['text']
#         # 提取每篇论文的详细信息
#         matches = metadata_pattern.findall(text)
#         id_set = set() # 论文去重
#         for i, match in enumerate(matches):
#             title, paper_id, authors, abstract = match
#             title = title.strip()
#             paper_id = paper_id.strip()
#             authors = authors.strip()
#             abstract = abstract.strip()
#             if paper_id not in id_set:
#                 paper_metadata.append(
#                     {
#                         "title": title,
#                         "ID": paper_id,
#                         "authors": authors,
#                         "abstract": abstract,
#                     }
#                 )
#             id_set.add(paper_id)
#     for i, metadata in enumerate(paper_metadata):
#         title = re.sub(invalid_chars, '_', metadata['title'])
#         ID = metadata['ID']
#         text = asyncio.run(
#             download_tool.ainvoke({
#                 "input": ID,
#             })
#         )[0]['text']
#         url = url_pattern.search(text).group(0).strip()
#         paper_metadata[i]['url'] = url
#         download(url, f'{PATH}{title}.pdf')
#         paper_localpath.append(f'{PATH}{title}')
#
#     return {'raw_papers_metadata': paper_metadata, 'raw_papers_localpath': paper_localpath}
#
# def sparse_node(state: PaperWritingState):
#     print(">>> 解析节点")
#     writer = get_stream_writer()
#     writer({'node': 'sparse_node'})
#
#     # PDF 2 Markdown
#     paper_metadata = state['raw_papers_metadata']
#     for metadata in paper_metadata:
#         title = re.sub(invalid_chars, '_', metadata['title'])
#         documents = PARSER.load_data(f"{PATH}{title}.pdf")
#         with open(f"{PATH}{title}.md", "w", encoding="utf-8") as f:
#             for doc in documents:
#                 f.write(doc.text + "\n\n")
#     # 论文摘要
#     prompt = '''
# 你是一个论文总结分析专家。
#
# *任务*
# 根据传入的论文，解析论文的摘要(abstract)、方法论(methodology)、结论(conclusion)。
#
# *定义*
# 摘要: 包括四个基本要素：研究目的、研究方法、研究结果和结论，并要求客观中立、概括性强、信息完整。200-300字。
# 方法论：包括使用的研究方法、模型以及创新点(如果有)。100-200字。
# 结论：论文最后得出的研究结论。100-200字。
#
# *输出要求*
# 严格遵循JSON的格式，不要有任何其他内容和其他形式的输出，输出的格式如下。
# {"abstract": "", "methodology": "", "conclusion":""}
# '''
#     parsed_papers =  []
#     for metadata in paper_metadata:
#         title = re.sub(invalid_chars, '_', metadata['title'])
#         author = metadata['authors']
#         year = metadata['year']
#         with open(f"{PATH}{title}.md", "r", encoding="utf-8") as f:
#             prompts = [
#                 {'role': 'system', 'content': prompt},
#                 {'role': 'user', 'content': f},
#             ]
#             response = LLM.invoke(prompts).content
#             json_response = {'title': title, 'author': author, 'year': year, 'abstract': '', "methodology": '', "conclusion": ''}
#             try:
#                 j = json.loads(response)
#                 for key, value in j.items():
#                     json_response[key.lower()] = value
#             except Exception as e:
#                 raise json.JSONDecodeError(f'plan node "JSON loads error": {e}:')
#             parsed_papers.append(json_response)
#
#
#     return {'parsed_papers': parsed_papers}
