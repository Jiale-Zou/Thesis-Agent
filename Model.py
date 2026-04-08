import json
keys = json.load(open("config.json", 'r'))

def deepseek_chat(temperature=0.7):
    from langchain_openai import ChatOpenAI
    API_KEY = keys['deepseek-key']
    # 使用langchain创建访问OpenAI的Model。
    model = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=API_KEY,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=temperature
    )
    return model


def qwen_chat(temperature=0.7):
    from langchain_openai import ChatOpenAI
    API_KEY = keys['qwen-key']
    # 使用langchain创建访问OpenAI的Model。
    model = ChatOpenAI(
        model="qwen-plus",
        openai_api_key=API_KEY,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature
    )
    return model

def qwen_code(temperature=0.7):
    from langchain_openai import ChatOpenAI
    API_KEY = keys['qwen-key']
    # 使用langchain创建访问OpenAI的Model。
    model = ChatOpenAI(
        model="qwen3-coder-next",
        openai_api_key=API_KEY,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature
    )
    return model


def chat_model(name='qwen', temperature=0.7):
    if name.lower() == 'qwen':
        return qwen_chat(temperature)
    elif name.lower() == 'deepseek':
        return deepseek_chat(temperature)
    else:
        raise ValueError(f'Unknown chat model: {name}')

def code_model(name='qwen', temperature=0.7):
    if name.lower() == 'qwen':
        return qwen_code(temperature)
    else:
        raise ValueError(f'Unknown chat model: {name}')

def parser(type="markdown"):
    import os
    from llama_parse import LlamaParse
    os.environ["LLAMA_CLOUD_API_KEY"] = keys['lamma-key']
    parser = LlamaParse(
        result_type=type,  # 可选 "text" 或 "markdown"
        verbose=True,
    )
    return parser
