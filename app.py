import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

st.set_page_config(page_title="InternLM2.5-Chat-20B-ecommerce-FT", page_icon="🦜🔗")
# st.title("InternLM2.5-Chat-20B-ecommerce-FT")

# 初始化模型
@st.cache_resource
def init_models():
    embed_model = HuggingFaceEmbedding(
        model_name="/root/model/sentence-transformer"
    )
    Settings.embed_model = embed_model

    # 定义系统提示
    system_prompt = """
    您是一位温柔、可爱、高情商的客服，您的任务是对顾客的问题总会耐心解答，给出让顾客满意的回答。
    请注意以下几点：
    1. 如果问题超出了提供的信息范围，请礼貌地表示您没有这方面的信息。
    2. 用“亲亲”“您”等词语拉近与顾客的距离。
    """
    llm = HuggingFaceLLM(
        model_name="/root/model/merged",
        tokenizer_name="/root/model/merged",
        model_kwargs={"trust_remote_code":True},
        tokenizer_kwargs={"trust_remote_code":True},
        system_prompt=system_prompt
    )
    Settings.llm = llm

    documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    return query_engine

# 检查是否需要初始化模型
if 'query_engine' not in st.session_state:
    st.session_state['query_engine'] = init_models()

def greet2(question):
    response = st.session_state['query_engine'].query(question)
    return response

      
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "您好，我是您的智能客服小若，有什么我可以帮助您的吗？"}]    

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "您好，我是您的智能客服小若，有什么我可以帮助您的吗？"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama_index_response(prompt_input):
    return greet2(prompt_input)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Gegenerate_llama_index_response last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama_index_response(prompt)
            # 将response转换为字符串
            response_str = str(response)
            # 提取第一个回答，忽略后续的额外问答
            first_answer = response_str.split("Query:")[0].strip()
            first_answer = first_answer.replace("<|endoftext|>","")
            placeholder = st.empty()
            placeholder.markdown(first_answer)
            
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)