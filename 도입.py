import os
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

import streamlit as st
import pandas as pd


# API KEY 정보로드
#load_dotenv()

# python -m streamlit run main.py
st.title("📚얘들아 물리 쉬워✨")


os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]


st.session_state.api_key = st.secrets["openai_api_key"]

# 풀이 후 풍선 표시 여부
if "balloons_t1_q1" not in st.session_state:
    st.session_state.balloons_t1_q1 = False
if "balloons_t1_q2" not in st.session_state:
    st.session_state.balloons_t1_q2 = False
if "balloons_t2_q1" not in st.session_state:
    st.session_state.balloons_t2_q1 = False
if "balloons_t2_q2" not in st.session_state:
    st.session_state.balloons_t2_q2 = False
if "balloons_t2_q3" not in st.session_state:
    st.session_state.balloons_t2_q3 = False


## 학생에게 api-key를 입력하게 할 경우
## ------(아래 주석을 해제해주세요)------
# api_key = st.text_input("🔑 새로운 OPENAI API Key", type="password")
# save_btn = st.button("설정 저장", key="save_btn")

# if save_btn:
#    settings.save_config({"api_key": api_key})
#    st.session_state.api_key = api_key
#    st.write("설정이 저장되었습니다.")
## --------------------------------

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 처음 1번만 실행하기 위한 코드
if "intro_1_tutor_messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["intro_1_tutor_messages"] = []

if "pdf_chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["pdf_chain"] = None

if "pdf_retriever" not in st.session_state:
    st.session_state["pdf_retriever"] = None


@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file_path):
    # Get filename without extension for the vectorstore
    file_base_name = os.path.splitext(os.path.basename(file_path))[0]
    vectorstore_path = f"./.cache/embeddings/{file_base_name}"
    
    # Check if we already have embeddings for this file
    if os.path.exists(vectorstore_path):
        # Load existing vectorstore
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["openai_api_key"]) #st.session_state.api_key)
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore.as_retriever()

    # ... existing document loading and splitting code ...
    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["openai_api_key"]) # st.session_state.api_key)

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    # Save the vectorstore
    vectorstore.save_local(vectorstore_path)
    
    return vectorstore.as_retriever()


# 체인 생성
def create_chain(retriever, prompt_path="prompts/doc-rag.yaml", model_name="gpt-4o"):
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


# 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
retriever = embed_file("textbooks/physics_textbook_trancated.pdf")
selected_model = "gpt-4o"
selected_prompt = "prompts/doc-rag.yaml"

rag_chain = create_chain(
    retriever, prompt_path=selected_prompt, model_name=selected_model
)
st.session_state["pdf_retriever"] = retriever
st.session_state["pdf_chain"] = rag_chain


# 사이드바 생성
with st.sidebar:

    st.text("AI튜터와 대화하기")
    messages = st.container(height=300)
        
    def print_messages():
        for chat_message in st.session_state["intro_1_tutor_messages"]:
           messages.chat_message(chat_message.role).write(chat_message.content)

    # 새로운 메시지를 추가
    def add_message(role, message):
        st.session_state["intro_1_tutor_messages"].append(ChatMessage(role=role, content=message))

    # 이전 대화 기록 출력
    print_messages()

    if user_input := st.chat_input("🤖 AI튜터에게 궁금한 내용을 물어보세요!"):

        chain = st.session_state["pdf_chain"]

        if chain is not None:
            # 사용자의 입력
            messages.chat_message("user").write(user_input)
            # 스트리밍 호출
            response = chain.stream(user_input)
            with messages.chat_message("assistant"):
                # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
                container = st.empty()

                ai_answer = ""
                for token in response:
                    ai_answer += token
                    container.markdown(ai_answer)

            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)


st.subheader("오늘 배운 물리 개념은?")
st.info("- 1.여러 가지 운동과 2.힘과 운동 소단원에서 배운 내용을 복습해봅시다:) \n - 아래 2개 탭을 모두 마치고 다음 단계로 넘어가주세요! \n - 다음 단계는 하단의 버튼과 함께 왼쪽 사이드바에서 직접 클릭할 수 있습니다. \n - 문제를 풀면서 모르는게 있으면 왼쪽 사이드바의 🤖AI 튜터에게 물어보세요!")

tab1, tab2 = st.tabs(["1. 여러 가지 운동", "2. 힘과 운동"])

with tab1:
    st.write("**Q1. 놀이공원에서 볼 수 있는 여러 가지 놀이 기구들의 운동 방향과 속력 변화를 표에 정리해봅시다.**")
    st.write("(셀을 더블클릭한 후 알맞은 설명을 골라주세요)")
    df = pd.DataFrame(
        [
            {"type": f"https://dimg.donga.com/ugc/CDB/SODA/Article/57/e0/f9/a4/57e0f9a4248cd2738de6.gif", "direction": None, "speed": None},
            {"type": f"https://mblogthumb-phinf.pstatic.net/MjAyNDAxMTJfMjM3/MDAxNzA1MDQxNTI4NTI5.H9ncgGIDHPaCqa0i3Mz-2s8NU-qvEye8YQwlSQaFWZUg.wDDf8kJoDCbWyuVEBEqkzxfxQCv5GnNAMYw0Bo4nZnIg.GIF.onnamong/743084218.gif?type=w800", "direction": None, "speed": None},
            {"type": f"https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDM2ZTRueWxqMW00emRyanIxaGVwdGx4amR2aW5scDNnYnMzbTg4YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WqSCrUOEWQpWC8JGNg/giphy.webp", "direction": None, "speed": None},
            {"type": f"https://mblogthumb-phinf.pstatic.net/MjAyMjEyMjhfMTMx/MDAxNjcyMTYzMDc2Mzc3.RACWbUSQ5ys66npAKl1ABuFkloP9bs3D3Hg6Hv0rhG0g.z4ZBQI_SRDXNCaO8v51EZrYxIkCwLcnawqYXJjqunrsg.GIF.mok5022/3546279707.gif?type=w800", "direction": None, "speed": None},
        ]
    )

    column_configuration = {
        "type": st.column_config.ImageColumn("놀이 기구 (더블클릭해서 확대해보세요)"),
        "direction": st.column_config.SelectboxColumn(
            "운동 방향", options=["변한다", "일정하다"]
        ),
        "speed": st.column_config.SelectboxColumn(
            "속력", options=["변한다", "일정하다"]
        ),
    }

    edited_df = st.data_editor(
        data=df,
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        ) # 👈 An editable dataframe

    st.session_state.q1 = False
    info_q1 = st.empty()
    if edited_df.notnull().all().all():
        if edited_df.iloc[0, 1] == "일정하다":
            if edited_df.iloc[0, 2] == "변한다":         
                if edited_df.iloc[1, 1] == "변한다":
                    if edited_df.iloc[1, 2] == "일정하다":
                        if edited_df.iloc[2, 1] == "변한다":
                            if edited_df.iloc[2, 2] == "변한다":
                                if edited_df.iloc[3, 1] == "변한다":
                                    if edited_df.iloc[3, 2] == "변한다":
                                        st.session_state.q1 = True
                                        info_q1.success("정답입니다!")
                                        if not st.session_state.balloons_t1_q1:
                                            st.balloons()
                                            st.session_state.balloons_t1_q1 = True
        if st.session_state.q1:
            info_q1.success("정답입니다!")
        else:
            info_q1.error("틀렸습니다. 다시 시도하세요.")

    st.write("")
    st.write("") 

    st.write("**Q2. 놀이 기구의 운동을 다음과 같이 분류해 봅시다.**")
    options_1 = st.multiselect(
        "속력만 변하는 운동",
        ["자이로드롭", "관람차", "롤러코스터", "바이킹"],
    )
    options_2 = st.multiselect(
        "운동 방향만 변하는 운동",
        ["자이로드롭", "관람차", "롤러코스터", "바이킹"],
    )
    options_3 = st.multiselect(
        "속력과 운동 방향이 모두 변하는 운동",
        ["자이로드롭", "관람차", "롤러코스터", "바이킹"],
    )

    st.session_state.q2 = False
    info_q2 = st.empty()
    if options_1 and options_2 and options_3:
        if options_1 == ["자이로드롭"]:
            if options_2 == ["관람차"]:
                if options_3 == ["롤러코스터", "바이킹"]:
                    st.session_state.q2 = True
                    info_q2.success("정답입니다!")
                    if not st.session_state.balloons_t1_q2:
                        st.balloons()
                        st.session_state.balloons_t1_q2 = True
        if st.session_state.q2:
            info_q2.success("정답입니다!")
        else:
            info_q2.error("틀렸습니다. 다시 시도하세요.")

    if st.session_state.q1 and st.session_state.q2:
        st.subheader("**핵심 정리!**")
        st.info("속력이 변하는 운동, 운동 방향이 변하는 운동, 속력과 운동 방향이 모두 변하는 운동은 모두 **가속도 운동**이다.")
        st.image("images/explain_1.png")
        st.image("images/explain_2.png")

# col1, col2 = st.columns(2)
# cont = st.container(border=True)

# with col1:
#     st.image("https://blog.kakaocdn.net/dn/dve48V/btqzx7xvXtM/lVxQZ8s7bY86RSZeVoCzc1/img.jpg", caption="이 운동의 이름을 맞춰라!")

# with col2:
#     st.image("https://i0.wp.com/imagine.gsfc.nasa.gov/features/yba/CygX1_mass/gravity/images/circular_motion_animation.gif?resize=350%2C350&ssl=1", caption="인공위성도 이 운동을 합니다.")
#     answer2 = st.radio("이 운동의 이름을 선택하세요", ["포물선 운동", "등속 원운동", "진자 운동"], key="answer2")
#     if st.button("제출", key="btn2"):
#         if answer2.strip().lower() == "등속 원운동":
#             st.success("정답입니다!")
#             st.balloons()
#             with cont:
#                 st.markdown(
#                     body="""
# # ## 등속 원운동

# # 운동 방향만 변하는 운동 놀이공원의 회전하는 관람차, 지구 주위를 도는 인공위성, 시계의 바늘 등은 일정한 속력으로 원을 그리며 운동하는데, 이러한 운동을 등속 원운동이라고 한다.
# # 등속 원운동 하는 물체는 속력이 변하지 않고 운동 방향만 변한다. (교과서 16쪽)
# """
#                 )
#         else:
#             st.error("틀렸습니다. 다시 시도하세요.")


# st.subheader("더 알아봅시다!")

# # 1. 탭 레이아웃 (수학 LaTeX, 유튜브, 데스모스 계산기)
# st.subheader("머리에 기름칠 하기")
# with st.container(border=True):
# #     st.text("아인슈타인은 머리가 복잡할 때 수학 문제를 풀면서 머리를 식혔다고 합니다.")
#     tab1, tab2 = st.tabs(["읽어보면 좋은 글", "관련 유튜브 영상"])

#     with tab1:
#         st.write("아래 기사를 참고해보세요. (기사 제목 누르기👇)")
#         st.page_link(page="https://blog.hyundai-rotem.co.kr/671", label="철도에 사용되는 과학기술: 고속 주행에도 안전한 커브는 OOO덕분?!", icon="📰")
#         st.image("https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc4OysX%2FbtruRMLUUsC%2FI2IEaLfd8YNZzB9p6Apzs0%2Fimg.jpg")

#         # 문제와 정답을 미리 설정 (LaTeX 수식 포함)
#         problems = {
#             "문제 1": r"12 + 8 = ?",
#             "문제 2": r"25 \div 5 = ?",
#             "문제 3": r"3 \times 7 = ?",
#             "문제 4": r"2x-1=3의 해는?"
#         }

#         # 정답 설정
#         answers = {
#             "문제 1": 20,
#             "문제 2": 5,
#             "문제 3": 21,
#             "문제 4": 2
#         }


#         # 사용자가 문제를 선택할 수 있도록 selectbox 추가
#         selected_problem_key = st.selectbox("풀고 싶은 문제를 선택하세요", list(problems.keys()))
        
#         # 세션 상태에 선택한 문제와 정답 저장
#         if selected_problem_key != st.session_state.get('selected_problem_key'):
#             st.session_state['selected_problem_key'] = selected_problem_key
#             st.session_state['correct_answer'] = answers[selected_problem_key]

#         # 세션 상태에서 문제와 정답 가져오기
#         selected_problem = problems[st.session_state['selected_problem_key']]
#         correct_answer = st.session_state['correct_answer']

#         # 문제 출력 (LaTeX 형식으로 수식 출력)
#         st.latex(rf"{selected_problem}")  # 수식 출력

#         # 답을 입력받기
#         user_answer = st.text_input("답을 입력하세요")

#         # spinner와 제출 버튼 생성 및 채점
#         if st.button("제출"):
#             with st.spinner('채점 중...'):
#                 if user_answer:
#                     try:
#                         if int(user_answer) == correct_answer:
#                             st.success("정답입니다!")
#                             st.balloons()  # 정답을 맞추면 풍선이 나타남
#                             # 문제를 초기화하여 새로운 문제를 풀 수 있도록 함
#                             del st.session_state['selected_problem_key']
#                         else:
#                             st.error("틀렸습니다. 다시 시도해보세요.")
#                     except ValueError:
#                         st.error("숫자를 입력해주세요.")
#                 else:
#                     st.error("답을 입력해주세요.")

    # with tab2:
    #     st.write("유튜브 영상으로 알아보는 오늘의 공부")
    #     st.video("https://youtu.be/FHrR_W4w_MA?feature=shared")
        # st.write("애니메이션으로 보는 시리즈(By Alan Becker)")
        # video = st.selectbox("강의 선택", ["애니메이션으로 보는 수학", "애니메이션으로 보는 물리학", "애니메이션으로 보는 기하학"])
        # if video == "애니메이션으로 보는 수학":
        #     st.video("https://www.youtube.com/watch?v=B1J6Ou4q8vE&list=PL7z8SQeih5Af9B2DshZul4KvTLI74NkUQ&index=1")
        # if video == "애니메이션으로 보는 물리학":
        #     st.video("https://youtu.be/ErMSHiQRnc8?list=PL7z8SQeih5Af9B2DshZul4KvTLI74NkUQ")
        # elif video == "애니메이션으로 보는 기하학":
        #     st.video("https://youtu.be/VEJWE6cpqw0?list=PL7z8SQeih5Af9B2DshZul4KvTLI74NkUQ")

    # with tab3:
    #     st.write("아래 계산기를 사용해보세요.")
    #     operation = st.selectbox("수학 연산 선택", ["과학용 계산기", "수학용 그래핑 계산기"])
    #     if operation == "과학용 계산기":
    #         st.components.v1.iframe("https://www.desmos.com/scientific", height=500)
    #     elif operation == "수학용 그래핑 계산기":
    #         st.components.v1.iframe("https://www.desmos.com/calculator", height=500)

with tab2:
    st.write("**Q1. 다음의 (괄호)안에 들어갈 표현으로 알맞은 과학 용어를 쓰세요.**")
    st.write("공을 발로 찰 때처럼 한 사람이 힘을 작용하기도 하지만 줄다리기처럼 여러 사람이 함께 힘을 작용할 때도 있다. 이처럼 한 물체에 여러 힘이 동시에 작용할 때 이 힘들과 같은 효과를 나타내는 하나의 힘을 1.(ㅇㅉㅎ)이라고 하며, 2.(ㅎㄹ)이라고도 한다.")
    t2_q1_1_input = st.text_input("1번 괄호의 답: ")
    t2_q1_2_input = st.text_input("2번 괄호의 답: ")

    if t2_q1_1_input and t2_q1_2_input:
        if t2_q1_1_input == "알짜힘" and t2_q1_2_input == "합력":
            st.success("정답입니다!")
            if not st.session_state.balloons_t2_q1:
                st.balloons()
                st.session_state.balloons_t2_q1 = True
            st.image("images/explain_3.png")
        else:
            st.error("틀렸습니다. 다시 시도하세요.")

    st.write("") 
    st.write("") 

    st.write("**Q2. 다음의 키워드를 이용해 뉴턴 운동 제1법칙을 설명해보세요.**")
    st.write("물체, 알짜힘, 운동 상태")
    t2_q2_input = st.text_area("뉴턴 운동 제1법칙이란: ")
    if t2_q2_input and st.button("제출", key="btn2"):
        with st.spinner("채점 중..."):
            client = OpenAI(api_key=st.session_state.api_key)
            system_prompt_q2 = """당신은 물리 교사입니다. 학생이 "물체, 알짜힘, 운동 상태" 키워드를 사용하여 뉴턴 운동 제1법칙을 설명하는 답안을 제출했습니다.

다음 기준에 따라 답안을 평가하고 점수와 피드백을 제공해주세요:

[평가 기준]
1. 필수 키워드 포함 (30점)
- "물체" 키워드 사용 (10점)
- "알짜힘" 키워드 사용 (10점)
- "운동 상태" 키워드 사용 (10점)

2. 뉴턴 제1법칙의 핵심 개념 설명 (70점)
- 알짜힘이 0인 조건 명시 (35점)
- 운동 상태 유지/변화 없음 설명 (35점)

[모범 답안 예시]
- "물체에 작용하는 알짜힘이 0일 때, 물체의 운동 상태가 변하지 않는다"
- "물체에 작용하는 알짜힘이 0일 때, 정지해 있는 물체는 계속 정지해 있고, 운동 중인 물체는 계속해서 등속도 운동을 한다"
- "물체에 작용하는 알짜힘이 0일 때, 물체가 현재의 운동 상태를 유지하려는 관성을 갖는다"

[출력 형식]
점수: [0-100점]

피드백: [2문장 이내로 부족한 부분이나 보완할 점을 구체적으로 제시]

예시 답변:
점수: 85점

피드백: 알짜힘이 0이라는 조건은 잘 설명했으나, 운동 상태가 변하지 않는다는 점을 더 명확히 표현하면 좋겠습니다. 모든 키워드를 자연스럽게 연결하여 하나의 완성된 문장으로 표현해보세요."""
            messages_q2 = [
                {"role": "system", "content": system_prompt_q2},
                {"role": "user", "content": t2_q2_input}
            ]
            response_q2 = client.chat.completions.create(model="gpt-4o", messages=messages_q2)
            msg_q2 = response_q2.choices[0].message.content
        st.info(msg_q2)
        score_q2 = re.search(r'(\d+)점', msg_q2).group(1)  # '100'
        if int(score_q2) >= 80:
            st.success("축하합니다!")
            if not st.session_state.balloons_t2_q2:
                st.balloons()
                st.session_state.balloons_t2_q2 = True
            st.image("images/explain_4.png")
        else:
            st.error("조금 더 분발해서 다시 시도하세요.")


    st.write("") 
    st.write("") 

    st.write("**Q3. 다음의 키워드를 이용해 뉴턴 운동 제2법칙을 설명해보세요.**")
    st.write("물체, 알짜힘, 가속도, 질량")
    t2_q3_input = st.text_area("뉴턴 운동 제2법칙이란: ")
    st.spinner("채점 중...")
    if t2_q3_input and st.button("제출", key="btn3"):
        with st.spinner("채점 중..."):
            client = OpenAI(api_key=st.session_state.api_key)
            system_prompt_q3 = """당신은 물리 교사입니다. 학생이 "물체, 알짜힘, 가속도, 질량" 키워드를 사용하여 뉴턴 운동 제2법칙을 설명하는 답안을 제출했습니다.

다음 기준에 따라 답안을 평가하고 점수와 피드백을 제공해주세요:

[평가 기준]
1. 필수 키워드 포함 (40점)
- "물체" 키워드 사용 (10점)
- "알짜힘" 키워드 사용 (10점)
- "가속도" 키워드 사용 (10점)
- "질량" 키워드 사용 (10점)

2. 뉴턴 제2법칙의 핵심 개념 설명 (60점)
- 알짜힘과 가속도의 비례 관계 설명 (30점)
- 질량과 가속도의 반비례 관계 설명 (30점)

[모범 답안 예시]
- "물체에 작용하는 알짜힘이 클수록 가속도가 크고, 물체의 질량이 클수록 가속도가 작아진다"
- "물체에 작용하는 알짜힘이 물체의 질량과 가속도의 곱과 같다"
- "물체의 가속도는 작용하는 알짜힘에 비례하고 질량에 반비례한다"

[출력 형식]
점수: [0-100점]

피드백: [2문장 이내로 부족한 부분이나 보완할 점을 구체적으로 제시]

예시 답변:
점수: 80점

피드백: 알짜힘과 가속도의 비례 관계는 잘 설명했으나, 질량과의 관계가 명확하지 않습니다. 물체의 질량이 가속도에 미치는 영향을 추가로 설명해보세요."""
            messages_q3 = [
                {"role": "system", "content": system_prompt_q3},
                {"role": "user", "content": t2_q3_input}
            ]
            response_q3 = client.chat.completions.create(model="gpt-4o", messages=messages_q3)
            msg_q3 = response_q3.choices[0].message.content
        st.info(msg_q3)
        score_q3 = re.search(r'(\d+)점', msg_q3).group(1)  # '100'
        if int(score_q3) >= 80:
            st.success("축하합니다!")
            if not st.session_state.balloons_t2_q3:
                st.balloons()
                st.session_state.balloons_t2_q3 = True
            st.image("images/explain_5.png")
        else:
            st.error("조금 더 분발해서 다시 시도하세요.")    


# Create columns with specific ratios
col1, col2, col3 = st.columns([5, 2, 3])

# Place the button in the last column

with col3:
    if st.button(
        label="다음단계로 넘어가기",
        icon="⏩",
        help="문제(P)로 넘어가기",
        type="primary"
    ):
        st.switch_page("pages/02_문제(P).py")