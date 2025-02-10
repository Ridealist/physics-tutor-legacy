import os
import streamlit as st

from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_teddynote.prompts import load_prompt
from langchain_teddynote import logging

# API KEY 정보로드
# config = settings.load_config()
# if "api_key" in config:
#     st.session_state.api_key = config["api_key"]
#     st.write(f'사용자 입력 API키 : {st.session_state.api_key[-5:]}')
# else : 
#     st.session_state.api_key = st.secrets["openai_api_key"]
#     st.write(f'API키 : {st.secrets["openai_api_key"][-5:]}')


st.session_state.api_key = st.secrets["openai_api_key"]


def init_setting():
    # 캐시 디렉토리 생성
    if not os.path.exists(".cache"):
        os.mkdir(".cache")

    # 파일 업로드 전용 폴더
    if not os.path.exists(".cache/files"):
        os.mkdir(".cache/files")

    if not os.path.exists(".cache/embeddings"):
        os.mkdir(".cache/embeddings")

    # 처음 1번만 실행하기 위한 코드
    if "messages" not in st.session_state:
        # 대화기록을 저장하기 위한 용도로 생성한다.
        st.session_state["messages"] = []

    if "pdf_chain" not in st.session_state:
        # 아무런 파일을 업로드 하지 않을 경우
        st.session_state["pdf_chain"] = None

    if "pdf_retriever" not in st.session_state:
        st.session_state["pdf_retriever"] = None



# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # # 모델 선택 메뉴
    # selected_model = st.selectbox(
    #     "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    # )

    # selected_prompt = st.selectbox(
    #     "프롬프트 선택",
    #     ["prompts/doc-rag.yaml", "prompts/doc-quiz.yaml", "prompts/doc-summary.yaml"],
    #     index=0,
    # )

    # update_btn = st.button("설정 업데이트")


# 모델 선택 메뉴
# st.selectbox(
#     "프롬프트 선택",
#     ["prompts/doc-rag.yaml", "prompts/doc-quiz.yaml", "prompts/doc-summary.yaml"],
#     index=0,
# )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key = st.session_state.api_key)

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


# 체인 생성
def create_chain(retriever, prompt_path="prompts/pdf-rag.yaml", model_name="gpt-4o"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt(prompt_path, encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key = st.session_state.api_key)
    
    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


col1, col2 = st.columns(2)
placeholder1 = col1.empty()
placeholder2 = col2.empty()


# if update_btn:
#     if st.session_state["pdf_retriever"] is not None:
#         retriever = st.session_state["pdf_retriever"]
#         chain = create_chain(
#             retriever, prompt_path=selected_prompt, model_name=selected_model
#         )
#         st.session_state["pdf_chain"] = chain
#     else:
#         retriever = embed_file(uploaded_file)
#         chain = create_chain(
#             retriever, prompt_path=selected_prompt, model_name=selected_model
#         )
#         st.session_state["pdf_retriever"] = retriever
#         st.session_state["pdf_chain"] = chain

# 이전 대화 기록 출력
# print_messages()


def init_pdf_chat(selected_prompt, selected_model):
    if uploaded_file:
        # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
        retriever = embed_file(uploaded_file)
        chain = create_chain(
            retriever, prompt_path=selected_prompt, model_name=selected_model
        )
        st.session_state["pdf_retriever"] = retriever
        st.session_state["pdf_chain"] = chain


# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

if user_input:
    chain = st.session_state["pdf_chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)

else:
    # 파일을 업로드 하라는 경고 메시지 출력
    warning_msg.error("파일을 업로드 해주세요.")
