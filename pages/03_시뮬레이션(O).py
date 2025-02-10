import re
import os
import sys
import streamlit as st
from typing import List
import getpass
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.globals import set_verbose
from langchain_core.messages.chat import ChatMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

import time
import importlib
import simulation.simulation as simulation

set_verbose(False)

st.session_state.api_key = st.secrets["openai_api_key"]

os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

if "submit_button_disabled" not in st.session_state:
    st.session_state["submit_button_disabled"] = True

if "tutor_messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["tutor_messages"] = []

DEFAULT_OPERATION = "자바 실험실"

def enalble_submit_button():
    st.session_state["submit_button_disabled"] = False

def disalble_submit_button():
    st.session_state["submit_button_disabled"] = True


def output_parser(response: str) -> str:
    content = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)

    if content:
        extracted_code = content.group(1)
        print(extracted_code)
        return extracted_code
    else:
        print("No Python code block found.")
        return None

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


st.title("물리 시뮬레이션 실험실 🧪")
st.info("- 시뮬레이션을 관찰하고 조작해보면서, 앞에서 제출한 답과 비교해보세요. \n - 시뮬레이션에 대해서 궁금한 사항은 사이드바의 🤖AI 튜터에게 질문해보세요. \n - 시뮬레이션을 조작하면서 알게 된 사실을 사이드바 하단 박스에 적어보세요.")

main_tab1 = st.container(border=True)
main_tab1.text("문제 상황")
main_tab1.image("images/problem_1.png")

# 모델 선택 메뉴
selected_model = "gpt-4o-mini"

default_operation = "자바 실험실"

# 사이드바 시스템 체인 생성
def generate_chain(model_name="gpt-4o-mini"):
    # 현재 선택된 operation 값 가져오기
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """당신은 친근하고 대화형 학습을 돕는 **물리 튜터**입니다.  
    **목표:** 학생이 '등속 원운동'에 대한 시뮬레이션에서 변수의 조작에 따른 물체의 운동 변화를 관찰하고, 이를 통해 자신의 추론을 검증해볼 수 있도록 지원하세요.  
    **중요:**  
    1. 절대로 '등속 원운동'에서 힘의 방향이나 크기에 대한 결론을 직접 알려주지 마세요.
       - 학생이 시뮬레이션을 통해 스스로 결과를 관찰하고 해석하도록 도와주세요.
       - 학생이 관찰을 서술하는 과정에서 편견 없이 관찰한 대로 기록하도록 유도하세요.
    2. 항상 **존댓말**로 대화하세요. 친절하고 격려하는 태도를 유지하며 학생이 부담 없이 실험과 기록을 진행하도록 돕습니다.  
    3. **답변 길이 제약:** 각 응답은 **2문장**을 넘지 않도록 간결하고 명확하게 작성하세요.

    **설명 전략:**
    1. 현재 선택되어 있는 {current_operation} 값에 따라 각각 다르게 안내하세요.
        if operation == "자바 실험실": 힘과 운동 방향의 각도를 조정하면서 원운동의 조건을 탐구해보도록 안내합니다.
        elif operation == "PhET": 행성의 위치나 크기를 조작해보면서, 어떨 때 행성이 원운동이 일어날 수 있을지 조작해보도록 안내합니다.
        elif operation == "자율실험실": 시뮬레이션 위에 있는 입력창에 내가 생각하는 힘의 방향이나 그 외 다양한 변수를 수정해보도록 안내합니다.

    **핵심 개념 질문 전략:**  
    1. 현재 선택되어 있는 {current_operation} 값에 따라 각각 다르게 질문하여 학생의 생각을 유도하세요.
        if operation == "자바 실험실": 힘의 방향과 물체의 운동 궤도 사이의 관계를 탐구해보도록 유도합니다.
        elif operation == "PhET": 행성의 위치나 크기를 조작해보면서, 중력이 무슨 역할을 하는지 탐구해보도록 유도합니다.
        elif operation == "자율실험실": 
            시뮬레이션 위에 있는 입력창에 어떻게 시뮬레이션을 수정하고 싶은지 작성할 수 있도록 예시와 같은 방법으로 학생의 생각을 유도합니다.
            예시:
                "원운동이 계속되려면 힘이 어떤 방향이어야 할 것 같나요?" 
                "'문제'에서 답한 내용을 반영한 시뮬레이션 보기 버튼을 눌러 확인해보세요."
                "시뮬레이션에 반영되어야 한다고 생각하는 내용을 시뮬레이션 위에 있는 입력창에 입력해주세요."
    """
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name,
        openai_api_key = st.session_state.api_key
    )
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history

# 사이드바 생성
with st.sidebar:

    options = ["자바 실험실", "자율실험실"]
    operation = st.pills("(어떤 시뮬레이션을 하고 있는지 선택해주세요)", options, selection_mode="single")

    if operation is None:
        operation = DEFAULT_OPERATION

    # 초기화 버튼 생성
    st.text("AI튜터와 대화하기")
    messages = st.container(height=300)
        
    def print_messages():
        for chat_message in st.session_state["tutor_messages"]:
           messages.chat_message(chat_message.role).write(chat_message.content)

    # 새로운 메시지를 추가
    def add_message(role, message):
        st.session_state["tutor_messages"].append(ChatMessage(role=role, content=message))

    # 이전 대화 기록 출력
    print_messages()

    if user_input := st.chat_input("🤖 AI튜터에게 궁금한 내용을 물어보세요!"):

        conv_chain = generate_chain(selected_model)

        # 사용자의 입력
        messages.chat_message("user").write(user_input)

        with messages.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()
            generator = conv_chain.stream(
                {
                    "input": user_input,
                    "current_operation": operation
                },
                config={"configurable": {"session_id": "ab12"}}
            )
            ai_answer = ""
            for token in generator:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)

    facts = st.text_area(
        label="관찰하면서 알게 된 것들을 적어봅시다.",
        placeholder="- 변수 조작에 따라 움직임이 달라진 점\n- 예측했던 것과 관찰한 결과가 다른점",
        height=200,
        on_change=enalble_submit_button
    )

    if not facts:
        disalble_submit_button()

    submit_button = st.button(
        label="제출하기",
        type="primary",
        use_container_width=True,
        disabled=st.session_state["submit_button_disabled"]
    )

    if submit_button:
        st.session_state["observation_user_facts"] = facts
        st.success("제출 완료!")


# 자율실험실 초기 세션 상태 설정
if 'button_timestamp' not in st.session_state:
    st.session_state.button_timestamp = time.time()
if 'last_input' not in st.session_state:
    st.session_state.last_input = ''

operation = "자바 실험실"

with st.container(border=True):
    tab1, tab2 = st.tabs(["자바 실험실", "자율실험실"])

    # Create columns with specific ratios
    col1, col2, col3 = st.columns([5, 2, 3])
    # Place the button in the last column
    with col1:
        if st.button(
            label="이전단계로 넘어가기",
            icon="⏪",
            help="문제(P)로 넘어가기",
            type="primary"
        ):
            st.switch_page("pages/02_문제(P).py")

    # Place the button in the last column
    with col3:
        if st.button(
            label="다음단계로 넘어가기",
            icon="⏩",
            help="돌아보기(E)로 넘어가기",
            type="primary"
        ):
            st.switch_page("pages/04_돌아보기(E).py")

    with tab1: 
        operation= "자바 실험실"
        # CSS 스타일 적용
        st.markdown("""
            <style>
            /* iframe 컨테이너 스타일링 */
            .iframe-container {
                width: 700px;
                height: 600px;
                overflow: hidden;
                position: relative;
            }
            
            /* iframe 자체 스타일링 */
            .iframe-container iframe {
                width: 720px;  /* 스크롤바 여유 공간 */
                height: 800px;
                border: none;
                position: absolute;
                top: -250px;  /* 상단 여백 조절 */
                left: 0;
                margin: 0;
                padding: 0;
            }
            </style>
        """, unsafe_allow_html=True)
        # HTML div로 감싸서 iframe 생성
        st.markdown("""
            <div class="iframe-container">
                <iframe 
                    src="https://javalab.org/condition_of_circular_movement/"
                    scrolling="no"
                    frameborder="0"
                ></iframe>
            </div>
        """, unsafe_allow_html=True)
    with tab2: 
        operation= "자율실험실"
        
        # 챗봇 설정
        #chat = ChatAnthropic(model="claude-3-5-sonnet-20240620")
        chat = ChatOpenAI(model="gpt-4o-mini", api_key=st.session_state.api_key)

        # 시뮬레이션 수정중에는 화면 보이지 않게 스위치
        fixingNow = False
        afterFixing = False

        # 원본 코드 읽기 
        if "original_code" not in st.session_state:
            if os.path.exists('./simulation/simulation_user.py') and st.session_state.last_input:
                with open('./simulation/simulation_user.py', 'r', encoding='utf-8') as file_user:
                    st.session_state.current_code = file_user.read()
            else:
                with open('./simulation/simulation.py', 'r', encoding='utf-8') as file_original:
                    st.session_state.original_code = file_original.read()
            st.session_state.current_code = st.session_state.original_code
        
        # 수정된 시뮬레이션이 있는 경우
        if os.path.exists('./simulation/simulation_user.py') and st.session_state.last_input:
            st.info("🔄 수정된 시뮬레이션이 실행 중입니다")
            
            # 사용자 입력 받기
            user_input = st.chat_input("🤖 시뮬레이션을 어떻게 더 수정하고 싶은지 설명해주세요")
            
            # 원본으로 돌아가기 버튼
            if st.button("원본 시뮬레이션으로 돌아가기"):
                if 'simulation_user' in sys.modules:
                    del sys.modules['simulation_user']
                os.remove('./simulation/simulation_user.py')
                st.session_state.current_code = st.session_state.original_code
                st.session_state.last_input = ''
                st.rerun()
                simulation.init_simulation_state()
            
            # 사용자가 입력한 경우
            if user_input and user_input != st.session_state.get('last_input', ''):
                fixingNow = True
                st.session_state.last_input = user_input
                
                with st.spinner("요청에 따라 시뮬레이션을 수정하고 있습니다..."):
                    # AI 응답 받기
                    full_prompt = f"""
                    현재 시뮬레이션 코드를 분석하고, 사용자의 요청에 따라 수정해주세요.
                    코드는 Python과 Streamlit을 사용하는 물리 시뮬레이션입니다.

                    현재 코드:
                    {st.session_state.current_code}

                    사용자가 현재 코드에 대해 매우 불만족할 경우에는 기존 코드를 참고하여 수정해주세요.

                    기존 코드:
                    {st.session_state.original_code}

                    사용자 요청: "{user_input}"

                    다음 사항들을 고려하여 수정해주세요:
                    1. 시뮬레이션의 물리적 특성 (속도, 힘, 궤도 등)
                    2. 시각적 요소 (색상, 크기, 벡터 표시 등)

                    응답 형식:
                    1. 수정이 필요한 함수나 클래스의 전체 코드를 ```python ``` 블록 안에 작성하세요
                    2. 여러 함수를 수정할 경우 각각 별도의 코드 블록으로 작성하세요
                    3. 수정된 부분에 대한 설명을 추가해주세요
                    """
                    response = chat.invoke([HumanMessage(content=full_prompt)])
                    
                    try:
                        # AI 응답 처리 및 코드 수정
                        modified_section = response.content
                        code_pattern = r'```python\n(.*?)```'
                        code_matches = re.findall(code_pattern, modified_section, re.DOTALL)
                        
                        if code_matches:
                            modified_code = st.session_state.current_code
                            
                            for code_block in code_matches:
                                match = re.search(r'(def|class)\s+(\w+)', code_block)
                                if match:
                                    target_name = match.group(2)
                                    pattern = rf'(def|class)\s+{target_name}[^\n]*\n(?:(?!def|class).*\n)*'
                                    modified_code = re.sub(pattern, code_block + '\n', modified_code)
                            
                            # 수정된 코드 저장
                            with open('./simulation/simulation_user.py', 'w', encoding='utf-8') as file:
                                file.write(modified_code)
                            
                            st.session_state.current_code = modified_code
                            fixingNow = False
                            st.rerun()
                        else:
                            st.error("AI 응답에서 코드 블록을 찾을 수 없습니다.")
                            
                    except Exception as e:
                        st.error(f"코드 수정 중 오류가 발생했습니다: {str(e)}")
            
            # 수정된 시뮬레이션 실행
            max_attempts = 5  # 최대 수정 시도 횟수 (AI가 코드를 잘못 수정해서 재수정하는 최대 횟수)
            attempt = 0
            success = False
            
            while not success and attempt < max_attempts:
                try:
                    if 'simulation_user' in sys.modules:
                        del sys.modules['simulation_user']
                    from simulation.simulation_user import main
                    st.divider()
                    main()
                    fixingNow = False
                    # 시뮬레이션 실행 전 컨테이너 초기화
                    success = True  # 성공적으로 실행됨
                    attempt = 0

                except Exception as e:
                    attempt += 1
                    error_msg = str(e)
                    st.error(f"시뮬레이션 실행 중 오류가 발생했습니다 (시도 {attempt}/{max_attempts}): {error_msg}")
                    
                    if attempt < max_attempts:  # 마지막 시도가 아닌 경우에만 수정 시도
                        with st.spinner(f"AI가 오류를 분석하고 코드를 수정하고 있습니다... (시도 {attempt}/{max_attempts})"):
                            error_prompt = f"""
                            이전에 수정한 시뮬레이션 코드에서 오류가 발생했습니다.
                            오류 내용: {error_msg}

                            현재 코드:
                            {st.session_state.current_code}

                            다음 사항들을 고려하여 수정해주세요:
                            1. 시뮬레이션의 물리적 특성 (속도, 힘, 위치 등)
                            2. 시각적 요소 (색상, 크기, 벡터 표시 등)

                            응답 형식:
                            1. 수정이 필요한 함수나 클래스의 전체 코드를 ```python ``` 블록 안에 작성하세요
                            2. 여러 함수를 수정할 경우 각각 별도의 코드 블록으로 작성하세요
                            3. 수정된 부분에 대한 설명을 추가해주세요
                            """
                            response = chat.invoke([HumanMessage(content=error_prompt)])
                            
                            try:
                                # AI 응답 처리 및 코드 수정
                                modified_section = response.content
                                code_pattern = r'```python\n(.*?)```'
                                code_matches = re.findall(code_pattern, modified_section, re.DOTALL)
                                
                                if code_matches:
                                    modified_code = st.session_state.current_code
                                    
                                    for code_block in code_matches:
                                        match = re.search(r'(def|class)\s+(\w+)', code_block)
                                        if match:
                                            target_name = match.group(2)
                                            pattern = rf'(def|class)\s+{target_name}[^\n]*\n(?:(?!def|class).*\n)*'
                                            modified_code = re.sub(pattern, code_block + '\n', modified_code)
                                            
                                    # 수정된 코드 저장
                                    with open('./simulation/simulation_user.py', 'w', encoding='utf-8') as file:
                                        file.write(modified_code)
                                    
                                    st.session_state.current_code = modified_code
                                    st.rerun()

                                else:
                                    st.error("AI 응답에서 코드 블록을 찾을 수 없습니다.")
                                    
                            except Exception as e:
                                st.error(f"코드 수정 중 오류가 발생했습니다: {str(e)}")

                    else:
                        st.error("최대 시도 횟수를 초과했습니다. 수정에 실패했습니다.")
        
        # 기본 시뮬레이션 실행 중인 경우
        else:
            st.success("✨ 기본 시뮬레이션이 실행 중입니다")

            user_input = st.chat_input("🤖아래 시뮬레이션을 어떻게 수정하고 싶은지 설명해주세요")
            
            # 사용자의 답변을 반영한 시뮬레이션 보기 버튼 추가
            if st.button("또는, '문제'에서 답한 내용을 시뮬레이션에 반영하기"):
                if "predict_user_reason" in st.session_state and "predict_user_drawing" in st.session_state:
                    user_input_already = f"""
                    사용자의 설명: {st.session_state['predict_user_reason']}
                    사용자의 그림 설명: {st.session_state['predict_user_drawing']}

                    위 내용을 바탕으로 시뮬레이션을 수정해주세요.
                    """
                    user_input = user_input_already
                    
                else:
                    st.warning("먼저 '문제' 페이지에서 답변을 입력해주세요.")
                    
            # 기본 시뮬레이션 실행
            if not user_input and not fixingNow:
                from simulation.simulation import main
                st.divider()
                main()
            
            # 사용자가 입력한 경우
            if user_input and user_input != st.session_state.get('last_input', ''):
                st.session_state.last_input = user_input
                fixingNow = True
                with st.spinner("요청에 따라 시뮬레이션을 수정하고 있습니다..."):
                    # AI 응답 받기
                    full_prompt = f"""
                    현재 시뮬레이션 코드를 분석하고, 사용자의 요청에 따라 수정해주세요.
                    코드는 Python과 Streamlit을 사용하는 물리 시뮬레이션입니다.

                    현재 코드:
                    {st.session_state.current_code}

                    사용자 요청: "{user_input}"

                    다음 사항들을 고려하여 수정해주세요:
                    1. 시뮬레이션의 물리적 특성 (속도, 힘, 궤도 등)
                    2. 시각적 요소 (색상, 크기, 벡터 표시 등)

                    응답 형식:
                    1. 수정이 필요한 함수나 클래스의 전체 코드를 ```python ``` 블록 안에 작성하세요
                    2. 여러 함수를 수정할 경우 각각 별도의 코드 블록으로 작성하세요
                    3. 수정된 부분에 대한 설명을 추가해주세요
                    """
                    response = chat.invoke([HumanMessage(content=full_prompt)])
                    
                    try:
                        # AI 응답 처리 및 코드 수정
                        modified_section = response.content
                        code_pattern = r'```python\n(.*?)```'
                        code_matches = re.findall(code_pattern, modified_section, re.DOTALL)
                        
                        if code_matches:
                            modified_code = st.session_state.current_code
                            
                            for code_block in code_matches:
                                match = re.search(r'(def|class)\s+(\w+)', code_block)
                                if match:
                                    target_name = match.group(2)
                                    pattern = rf'(def|class)\s+{target_name}[^\n]*\n(?:(?!def|class).*\n)*'
                                    modified_code = re.sub(pattern, code_block + '\n', modified_code)
                            
                            # 수정된 코드 저장
                            with open('./simulation/simulation_user.py', 'w', encoding='utf-8') as file:
                                file.write(modified_code)
                            
                            st.session_state.current_code = modified_code
                            fixingNow = False
                            st.rerun()
                        else:
                            st.error("AI 응답에서 코드 블록을 찾을 수 없습니다.")
                            
                    except Exception as e:
                        st.error(f"코드 수정 중 오류가 발생했습니다: {str(e)}")
# operation == "PhET": st.components.v1.iframe("https://phet.colorado.edu/sims/html/my-solar-system/latest/my-solar-system_all.html?locale=ko", height=800, width=1000)
