import os
import streamlit as st

from langchain_core.messages.chat import ChatMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI #, OpenAIEmbeddings

from langchain_teddynote import logging

st.session_state.api_key = st.secrets["openai_api_key"]

os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 세션 기록을 저장할 딕셔너리
store = {} 


st.title("배운 내용 적용하기 📌")
st.info("- 배운 내용을 새로운 상황에 적용해보며 개념을 더 깊게 이해해봅시다. \n - 지구 주위를 도는 달의 모습을 관찰해보고 그 이유를 설명해보세요.")

st.image("https://bobmoler.wordpress.com/wp-content/uploads/2019/03/orbit_360p30-1.gif", caption="지구 주위를 도는 달의 모습")

# textbook_container = st.empty()
with st.container(border=True):  # border=True로 실선 테두리 추가
    st.markdown(body="""
    ## 등속 원운동

    운동 방향만 변하는 운동 놀이공원의 회전하는 관람차, 지구 주위를 도는 인공위성, 시계의 바늘 등은 일정한 속력으로 원을 그리며 운동하는데, 이러한 운동을 등속 원운동이라고 한다.
    등속 원운동 하는 물체는 속력이 변하지 않고 운동 방향만 변한다. (교과서 16쪽)
    """)

# 처음 1번만 실행하기 위한 코드
if "messages_application" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages_application"] = []


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages_application"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages_application"].append(ChatMessage(role=role, content=message))

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환



# 적용 질문(나중엔 DB에서 불러오면 좋을 듯)
applyingQuestion = "위 그림과 같이 달이 공전 궤도를 유지하면서 계속 지구를 돌 수 있는 이유는 무엇일까요?" 

# 적용하기 모범답안 (나중엔 DB에서 불러오면 좋을 듯)
applyingModeledAnswer = "달에는 지구가 작용한 중력이 계속해서 지구 중심방향으로 작용하고 있으며, 이 중력이 달의 공전 궤도를 유지할 수 있도록 구심력의 역할을 하여 달은 지구 주변을 계속 돌 수 있습니다." 

# 답변 비교 점수
def relevance_check(applyingModeledAnswer, model_name="gpt-4o"):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grader assessing the relevance of a student's essay answer to a given model answer. 
        Here is the model answer: {applyingModeledAnswer}

        Consider both the student's current answer and their previous responses in the conversation history to evaluate their understanding.
        
        Evaluation criteria:
        1. Accuracy and relevance compared to the model answer
        2. Progressive improvement in understanding shown through the conversation
        3. Integration of previous responses with new insights
        
        Combine the current input and conversation history to form a complete understanding of the student's answer.
        Then compare this combined answer against the model answer.
        
        Calculate a precise percentage score from 0 to 100 based on the above criteria.
        Be detailed in your scoring - use decimal points if needed (e.g., 67.5, 82.3, etc.).
        Do not round to the nearest 5 or 10.
        
        Return only the number without any additional text or explanation.""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("ai", "Based on both the current answer and previous responses, calculate a precise score from 0-100 with decimal points if needed."),
        ]
    )

    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key = st.session_state.api_key)

    with_message_history = RunnableWithMessageHistory(
        prompt | llm,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    chain = with_message_history | StrOutputParser()
    
    # 체인을 실행하여 실제 결과값을 받아옴
    score = chain.invoke(
        {"input": user_input, "applyingModeledAnswer": applyingModeledAnswer},
        config={"configurable": {"session_id": "abc123"}}
    )
    
    # 문자열에서 숫자만 추출
    try:
        # 마지막 숫자를 찾아 반환
        import re
        numbers = re.findall(r'\d+\.?\d*', score)
        if numbers:
            return numbers[-1]  # 마지막 숫자 반환
        else:
            return "0"  # 숫자를 찾지 못한 경우
    except Exception as e:
        print(f"Error parsing score: {e}")
        return "0"

# 답변 체인 생성
def create_chain(model_name="gpt-4o"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 친근하고 대화형 학습을 돕는 **물리 튜터**입니다.  
**목표:** 학생이 새로 익힌 '등속 원운동'과 관련된 개념을 실제 문제 상황에 적용하고, 이를 통해 개념의 유용성을 탐구하며 반성적 사고를 할 수 있도록 돕습니다.

**중요:**  
1. 학생이 새로 익힌 개념을 문제 풀이에 적용하도록 격려하세요.  
2. 학생이 자신의 답변을 성찰하며 새로운 개념이 기존 선개념과 어떻게 다른지 스스로 탐색할 수 있도록 질문을 유도하세요.  
3. 직접적인 정답을 제공하지 말고, 학생이 스스로 개념을 활용하여 답을 구성하도록 도와주세요.  
4. 항상 존댓말을 사용하여 일관된 대화를 유지하세요.  
5. 두 문장을 넘지 않게 대화를 생성하세요.
6. 학생의 답변에 오개념이 있다면 오개념에 불만족을 가질 수 있도록 반례를 제시하여 학생과 문답하세요. 만약 3회 이상 문답했지만 계속 오개념을 고수하는 경우에는 과학적 개념을 알려주세요. 다음과 같은 오개념이 있을 수 있습니다.
    1) 운동 방향으로 힘이 작용해야 한다고 생각하는 경우, 힘이 작용하지 않는 경우를 제시
    2) 원심력이 실제 작용하는 힘이라고 생각하는 경우, 원심력은 누가 어떻게 작용한 것인지 생각해보도록 제시

**대화 스타일:**  
- 학생이 새로 익힌 개념을 활용하여 자신감을 가질 수 있도록 긍정적이고 격려하는 태도로 대화하세요.  
- 열린 질문을 통해 학생이 자신의 답변을 스스로 검토하고 반성하도록 유도하세요.  
- 학생이 익힌 개념이 문제 상황에서 어떻게 적용되는지 명확히 이해할 수 있도록 돕되, 지나친 암시를 주지 마세요.

**핵심 질문 유도:**  
- "달이 일정한 궤도를 따라 움직이기 위해 어떤 힘이 필요할까요?"
- "달이 일정한 궤도를 따라 움직이기 위한 힘은 어떤 방향으로 작용해야 할까요?"
- "달에 작용하는 힘은 누가 어떻게 작용한 건가요?
- "만약 달에 작용하는 힘이 없다면, 달은 어떤 방향으로 움직일 것이라고 생각하시나요?"  
- "지금까지 배운 개념을 활용해서 달의 운동을 설명해 주시겠어요?"  

**중요 개념을 유도하는 과정:**  
- 학생이 '등속 원운동'의 개념을 달의 운동에 적용하도록 유도하세요.  
- 달의 공전에 대해 논의하면서, 구심력(지구가 작용하는 중력)이 달의 운동을 유지하는 데 필요한 이유를 학생이 스스로 정리할 수 있도록 돕습니다.  
- 필요하다면 실생활의 유사한 예를 제시하여 학생이 개념을 명확히 할 수 있도록 지원하세요.  

**대화 마무리:**  
학생이 자신의 답변을 성찰하도록 돕고, 새로 익힌 개념이 문제 풀이에 어떻게 적용되었는지 반성적으로 생각하게 하세요.  
- 예: "지금까지 문답한 내용을 바탕으로, 달이 공전궤도를 계속 유지하면서 돌 수 있는 이유를 다시 한 번 정리하여 설명해 봅시다.""",
            ),
            # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key = st.session_state.api_key)
    
    # 단계 8: 체인(Chain) 생성
    with_message_history = (
        RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
            prompt | llm,
            get_session_history,  # 세션 기록을 가져오는 함수
            input_messages_key="input",  # 입력 메시지의 키
            history_messages_key="history",  # 기록 메시지의 키
        )
    )
    chain = with_message_history | StrOutputParser()

    return chain


# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("🤖 답변을 작성해주거나, AI튜터에게 궁금한 내용을 물어보세요!")

# 체인 생성
if "application_chain" not in st.session_state:
    st.session_state["application_chain"] = create_chain()

chain = st.session_state["application_chain"]

# 사이드바에 relevance score를 표시할 컨테이너 생성
if "relevance_score" not in st.session_state:
    st.session_state["relevance_score"] = 0

# 사이드바에 점수 표시
with st.sidebar:
    st.header("모범 답안과의 일치율")
    # 점수 표시를 위한 컨테이너들을 미리 생성
    score_container = st.empty()
    score_text_container = st.empty()

if len(st.session_state["messages_application"]) == 0:
    init_user_input = f"""다음의 "대화 시작 제안" 중 하나의 질문으로 대화를 시작해줘. 나는 그 대화에 맞춰서 너가 낸 문제를 해결해볼게.  
다른 얘기를 하지 말고 오로지 '질문'만 제시하면서 너의 대화를 마무리해줘. 다시 한번 얘기하지만 처음에는 '질문'만 얘기하는 거야 다른 내용은 전혀 없이.

**대화 시작 제안:**  
- {applyingQuestion}  
  오늘 학습한 개념을 활용해서 설명해 봅시다.
- 방금 학습한 개념을 바탕으로, 다음 질문에 대답해봅시다.  
  {applyingQuestion}"""
    
    response = chain.stream(
        {"input": init_user_input},
        # 설정 정보로 세션 ID "abc123"을 전달합니다.
        config={"configurable": {"session_id": "abc123"}},
    )
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
        
    # 대화기록을 저장한다.
    add_message("assistant", ai_answer)

else:
    if user_input:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(
            {"input": user_input},
            # 설정 정보로 세션 ID "abc123"을 전달합니다.
            config={"configurable": {"session_id": "abc123"}},
        )
        
        # relevance score 계산 및 업데이트
        try:
            score = float(relevance_check(applyingModeledAnswer))
            st.session_state["relevance_score"] = score
            
            # 점수에 따라 다른 색상의 프로그레스 바 표시
            if score >= 80:
                st.markdown("""
                    <style>
                        .stProgress > div > div {
                            background-color: blue;
                        }
                    </style>""", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown("""
                    <style>
                        .stProgress > div > div {
                            background-color: red;
                        }
                    </style>""", 
                    unsafe_allow_html=True
                )
            
            score_container.progress(score / 100)
            score_text_container.write(f"🎯 {score:.1f}점")
            
            # 80% 이상 달성 시 축하 효과와 마무리 메시지
            if score >= 80:
                st.balloons()
                with st.chat_message("assistant"):
                    st.write("🎉 축하합니다! 충분히 잘 이해하셨네요. 이제 다음 단계로 넘어가실 수 있습니다.")
                    st.subheader("더 알아봅시다!")

                    # # 1. 탭 레이아웃 (수학 LaTeX, 유튜브, 데스모스 계산기)
                    # st.subheader("머리에 기름칠 하기")
                    with st.container(border=True):
                    #     st.text("아인슈타인은 머리가 복잡할 때 수학 문제를 풀면서 머리를 식혔다고 합니다.")
                        tab1, tab2 = st.tabs(["읽어보면 좋은 글", "관련 유튜브 영상"])

                        with tab1:
                            st.write("아래 기사를 참고해보세요. (기사 제목 누르기👇)")
                            st.page_link(page="https://blog.hyundai-rotem.co.kr/671", label="철도에 사용되는 과학기술: 고속 주행에도 안전한 커브는 OOO덕분?!", icon="📰")
                            st.image("https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc4OysX%2FbtruRMLUUsC%2FI2IEaLfd8YNZzB9p6Apzs0%2Fimg.jpg")

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

                        with tab2:
                            st.write("유튜브 영상으로 알아보는 오늘의 공부")
                            st.video("https://youtu.be/FHrR_W4w_MA?feature=shared")
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

                # 대화 종료를 위한 플래그 설정
                st.session_state["conversation_completed"] = True
            
        except (ValueError, TypeError) as e:
            st.error(f"점수 계산 중 오류가 발생했습니다: {e}")
        
        # 대화가 완료되지 않은 경우에만 AI 응답 생성
        if not st.session_state.get("conversation_completed", False):
            # 스트리밍 호출
            response = chain.stream(
                {"input": user_input},
                config={"configurable": {"session_id": "abc123"}},
            )
            
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                for token in response:
                    ai_answer += token
                    container.markdown(ai_answer)
            
            # AI 답변도 대화기록에 저장
            add_message("user", user_input)
            add_message("assistant", ai_answer)


# Create columns with specific ratios
col1, col2, col3 = st.columns([5, 2, 3])

# Place the button in the last column
with col1:
    if st.button(
        label="이전단계로 넘어가기",
        icon="⏪",
        help="돌아보기(E)로 넘어가기",
        type="primary"
    ):
        st.switch_page("pages/04_돌아보기(E).py")