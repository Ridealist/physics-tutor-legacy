import streamlit as st
import io
import os

from PIL import Image
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI

from modules.multimodal import MultiModalwithHistory

st.session_state.api_key = st.secrets["openai_api_key"]

os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

st.session_state["init_user_input"] = False

if "submit_button_disabled" not in st.session_state:
    st.session_state["submit_button_disabled"] = True

# 처음 1번만 실행하기 위한 코드
if "messages_explanation" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages_explanation"] = []

if "show_answer" not in st.session_state:
    st.session_state["show_answer"] = False

if "multimodal_chain" not in st.session_state:
    st.session_state["multimodal_chain"] = None

# def save_draw_image(numpy_array):
#     image = Image.fromarray(numpy_array)
#     current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     file_name = f"e-drawing-{current_time}"
#     image.save(f"draw_images/{file_name}.png", format="PNG")
#     # return st.info("이미지 저장됨")

def enalble_submit_button():
    st.session_state["submit_button_disabled"] = False

def disalble_submit_button():
    st.session_state["submit_button_disabled"] = True


st.title("문제 돌아보기 ✅")
st.info("- 왼쪽 캔버스에 최종적으로 생각한 **(1)정답**과 함께 그렇게 생각한 **(2)이유**를 적어주세요. \n - 정답을 적으면 하단 오른쪽 '문제 정답' 탭에 채점 결과와 피드백이 나옵니다. \n - AI 튜터와 대화하면서 내가 잘못 알고 있던 개념은 없는지 파악하고 내 이해도를 더 정교화해보세요.")

tab1, tab2 = st.tabs(["문제 상황", "문제 정답"])

with tab1:
    st.subheader("문제 상황")
    st.image("images/problem_1.png")

with tab2:
    togle = st.empty()
    if "explanation_user_drawing" in st.session_state and "explanation_user_reason" in st.session_state:
        togle.empty()
        st.subheader("문제 정답")
        st.image("images/review_1.png")
        # st.write(st.session_state)
        st.session_state["show_answer"] = True
    else:
        togle.error("답변을 먼저 제출해주세요!")
        pass

# 사이드바 생성
with st.sidebar:

    st.text("최종적으로 문제를 해결해보세요.")
    st.text("(아래 캔버스에 생각한 내용을 그려보세요🎨)")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color="#000000",
        background_color="#EEEEEE",
        background_image=None,
        update_streamlit=True,
        height=400,
        width=400,
        drawing_mode="freedraw",
        point_display_radius=0,
        key="canvas",
    )

    reason = st.text_area(
        label="왜 그렇게 생각했는지 자세히 적어보세요.",
        height=200,
        on_change=enalble_submit_button
    )

    if not reason:
        disalble_submit_button()

    submit_button = st.button(
        label="제출하기",
        type="primary",
        use_container_width=True,
        disabled=st.session_state["submit_button_disabled"]
    )

    if submit_button:
        st.session_state["explanation_user_drawing"] = canvas_result.image_data
        st.session_state["explanation_user_reason"] = reason
        st.success("제출 완료!")

with tab2:
    if "explanation_user_drawing" in st.session_state and "explanation_user_reason" in st.session_state:
        if not st.session_state["show_answer"]:
            togle.empty()
            st.subheader("문제 정답")
            st.image("images/review_1.png")
            # st.write(st.session_state)
            st.session_state["show_answer"] = True
        else:
            pass


# 모델 선택 메뉴
selected_model = "gpt-4o-mini" # st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)

# 시스템 프롬프트 추가
system_prompt = """당신은 친절하고 대화형 학습을 돕는 **물리 튜터**입니다.  
**목표:** 학생이 '등속 원운동'에서 힘의 방향이 물체의 운동 궤적에 어떻게 작용하는지를 이해하도록 돕는 것입니다. 정답 화면을 제공한 후, 학생이 정답을 논리적으로 이해하고 설명할 수 있도록 대화형 질문과 설명을 진행하세요.  
**중요:**  
1. 정답을 단순히 제시하는 데 그치지 말고, 학생이 그 이유를 이해하고 자신의 언어로 정리할 수 있도록 돕습니다.  
2. 학생의 관점과 질문을 존중하며, 친근한 존댓말로 대화하세요.  
3. 각 응답은 간결하고 명확하게 작성하세요(2문단 이내).  
4. 학생이 이해하지 못한 부분이 있을 경우, 추가적인 설명과 예시를 활용해 개념을 명확히 전달하세요.

**프롬프트 시작 제안:**  
(정답 화면을 학생에게 보여준 후 대화를 시작한다고 생각하세요.)
1. **정답 화면 제시 후**  
- "이 화면에서 물체가 등속 원운동을 유지하기 위해 힘이 어떤 방향으로 작용하고 있는지 보여주고 있어요. 여기서 힘의 방향이 원의 중심을 향하는 것을 확인할 수 있습니다."  
2. **첫 질문:**  
- "이 화면에서 힘이 왜 원의 중심을 향해야 하는지 생각해 보셨나요? 지금까지의 관찰과 그림을 바탕으로 한 번 설명해 보세요."  

**대화 전략:**  
1. 학생의 답변을 기반으로 질문을 이어가며 올바른 개념을 도출하도록 돕습니다.  
2. 대화의 초점은 학생이 '등속 원운동에서 속도의 방향이 계속 바뀌기 때문에 중심 방향의 힘(구심력)이 필요하다'는 점을 논리적으로 이해하는 데 맞춰야 합니다.  
3. 정답 화면을 참고하며 학생의 추론 과정을 구체화하도록 질문을 활용하세요.

**핵심 질문 유도:**  
- "등속 원운동에서는 물체의 속력이 일정하지만, 속도의 방향은 계속 변합니다. 속도의 방향을 바꾸기 위해 힘이 어떤 역할을 한다고 생각하나요?"  
- "이 정답 화면에서 보이는 힘의 방향이 속도와 직각을 이루는 이유는 무엇일까요?"  
- "만약 힘이 중심을 향하지 않는다면, 물체의 운동 궤적은 어떻게 될까요?"  

**추가적인 설명 제공:**  
학생이 답을 이해하지 못하거나 잘못된 추론을 제시했을 경우, 올바른 개념을 설명하세요.  
- "등속 원운동은 물체가 일정한 속력으로 원 궤도를 따라 움직이는 운동이에요. 하지만 속도의 방향이 계속 바뀌기 때문에 가속도가 존재하게 됩니다. 이 가속도를 만들어내기 위해서는 항상 원의 중심 방향으로 힘이 작용해야 해요. 이를 구심력이라고 합니다."  
- "만약 힘이 중심 방향으로 작용하지 않는다면, 물체는 직선으로 운동을 계속하려는 경향(관성)에 의해 원 궤도를 벗어나게 됩니다."  

**대화 마무리:**  
학생이 자신의 언어로 개념을 정리할 수 있도록 유도하세요.  
- "지금까지 배운 내용을 바탕으로 정리해 볼까요? 물체가 등속 원운동을 유지하기 위해 힘의 방향이 왜 원의 중심을 향해야 하는지 설명해 주세요."  
- "정답 화면을 다시 보면서 힘의 방향과 운동 궤적의 관계를 한 번 더 정리해 보세요. 왜 등속 원운동에서 중심 방향의 힘이 필요한지, 그것이 어떻게 운동을 유지하는지 설명할 수 있나요?"  

**중요 사항:**  
- 학생이 스스로 설명하려고 할 때 격려하며, 잘못된 표현이 있어도 바로 정정하지 말고 추가 질문으로 사고를 확장하도록 유도하세요.  
- 정답 화면과 학생의 추론 과정을 연결시키는 데 집중하며, 이해가 완벽해질 때까지 인내심을 가지고 돕습니다."""


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages_explanation"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages_explanation"].append(ChatMessage(role=role, content=message))


# Function to convert PNG to BytesIO
def png_to_bytesio(ndarray_file):
    # Open the PNG image
    with Image.fromarray(ndarray_file) as img:
        # Create a BytesIO object
        byte_io = io.BytesIO()
        # Save the image into the BytesIO object as PNG
        img.save(byte_io, format='PNG')
        # Reset the file pointer to the beginning
        byte_io.seek(0)
        return byte_io


# 이미지을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_imagefile(ndarray_file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = png_to_bytesio(ndarray_file).getvalue()
    # file_content = file.read()
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"explain-drawing-{current_time}"
    # file_path = f"./.cache/files/{file.name}"
    file_path = f"./.cache/files/{file_name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


# 체인 생성
def init_multimodal_chain(system_prompt, user_prompt=None, model_name="gpt-4o"):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name,
        openai_api_key = st.session_state.api_key
    )
    # 멀티모달 객체 생성
    multimodal = MultiModalwithHistory(llm, system_prompt=system_prompt, user_prompt=user_prompt)
    st.session_state["multimodal_chain"] = multimodal


# 체인 생성
# def create_chain(prompt_filepath, task=""):
#     # prompt 적용
#     prompt = load_prompt(prompt_filepath, encoding="utf-8")

#     # 추가 파라미터가 있으면 추가
#     if task:
#         prompt = prompt.partial(task=task)

#     # GPT
#     #llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
#     llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=st.session_state.api_key)

#     # 출력 파서
#     output_parser = StrOutputParser()

#     # 체인 생성
#     chain = prompt | llm | output_parser
#     return chain

if "explanation_user_drawing" in st.session_state and "explanation_user_reason" in st.session_state:
    
    init_multimodal_chain(
        system_prompt=system_prompt,
        model_name="gpt-4o"
    )
    
    # 이전 대화 기록 출력
    print_messages()

    # 사용자의 입력
    user_input = st.chat_input("궁금한 내용을 물어보세요!")
    image_filepath = process_imagefile(st.session_state["explanation_user_drawing"])

    # 멀티모달 객체 생성
    multimodal = st.session_state["multimodal_chain"]

    if len(st.session_state["messages_explanation"]) == 0:
        init_user_input = f"이 그림은 '어떤 물체가 등속 원운동을 하기 위해서 물체에 어떤 방향으로 힘이 작용해야하는가'문제에 대해서 내가 생각한 힘의 방향을 그린 그림이야. 그리고 이렇게 그린 이유는 다음과 같아: {st.session_state["explanation_user_reason"]}."
        # 답변 요청
        response = multimodal.stream(
            user_prompt=init_user_input,
            image_url=image_filepath,
        )
        
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("assistant", ai_answer)

    else:   
        if user_input:
            # 답변 요청
            response = multimodal.stream(user_prompt=user_input)

            # 사용자의 입력
            st.chat_message("user").write(user_input)

            with st.chat_message("assistant"):
                # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
                container = st.empty()

                ai_answer = ""
                for token in response:
                    ai_answer += token.content
                    container.markdown(ai_answer)
                
            multimodal.add_messages(
                role="ai",
                content=ai_answer
            )

            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)


# Create columns with specific ratios
col1, col2, col3 = st.columns([5, 2, 3])

# Place the button in the last column
with col1:
    if st.button(
        label="이전단계로 넘어가기",
        key="prev_button",
        icon="⏪",
        help="시뮬레이션(O)으로 넘어가기",
        type="primary"
    ):
        st.switch_page("pages/03_시뮬레이션(O).py")

# Place the button in the last column
with col3:
    if st.button(
        label="다음단계로 넘어가기",
        key="next_button",
        icon="⏩",
        help="적용하기(A)로 넘어가기",
        type="primary"
    ):
        st.switch_page("pages/05_적용하기(A).py")
