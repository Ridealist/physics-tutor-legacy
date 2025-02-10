# 시뮬레이션 코드
import os
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math
from datetime import datetime
import time

# 파일 상단에 추가
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
ORBIT_RADIUS = 100
BALL_MASS = 1
BALL_RADIUS = 10
INITIAL_SPEED = 200
VECTOR_SCALE = 0.15
FPS = 60
center_x = WINDOW_WIDTH // 2
center_y = WINDOW_HEIGHT // 2

def init_simulation_state():
    if 'simulation_state' not in st.session_state:
        
        st.session_state.simulation_state = {
            'position': [center_x + ORBIT_RADIUS, center_y],
            'velocity': [0, INITIAL_SPEED],
            'trajectory_points': [],
            'simulation_running': False,
            'show_velocity_vector': True,
            'show_force_vector': False,
        }

# 힘 계산 함수.
def calculate_force(position, velocity):
    center = np.array([WINDOW_WIDTH/2, WINDOW_HEIGHT/2])
    pos = np.array(position)
    r = pos - center
    distance = np.linalg.norm(r)
    centripetal_force_magnitude = -BALL_MASS * INITIAL_SPEED**2 / distance
    centripetal_force_vector = centripetal_force_magnitude * r / distance # 구심력 크기에 방향 벡터 곱해준거임
    resultant_force = centripetal_force_vector
    
    return resultant_force

# 위치 업데이트 함수.
def update_position(dt):
    state = st.session_state.simulation_state
    force = calculate_force(state['position'], state['velocity'])
    acceleration = force / BALL_MASS
    
    # 벡터 연산으로 단순화
    state['velocity'] = [v + a * dt for v, a in zip(state['velocity'], acceleration)]
    state['position'] = [p + v * dt for p, v in zip(state['position'], state['velocity'])]
    
    # 궤적 점 개수 제한 (예: 최대 50개)
    MAX_TRAJECTORY_POINTS = 500
    state['trajectory_points'].append(state['position'].copy())
    if len(state['trajectory_points']) > MAX_TRAJECTORY_POINTS:
        state['trajectory_points'].pop(0)  # 가장 오래된 점 제거

# 벡터 표시 함수.
def create_arrow(start, end, color):
    # 리스트를 NumPy 배열로 변환
    start = np.array(start)
    end = np.array(end)
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    length = np.linalg.norm(end - start)
    if np.all(length == 0):
        return None
        
    unit_vector = np.array([dx, dy]) / length
    arrow_head_length = 20
    angle = np.radians(30)
    
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
    
    head1 = end - arrow_head_length * (rot_matrix @ unit_vector)
    head2 = end - arrow_head_length * (rot_matrix.T @ unit_vector)
    
    arrow_x = [start[0], end[0], None, end[0], head1[0], None, end[0], head2[0]]
    arrow_y = [start[1], end[1], None, end[1], head1[1], None, end[1], head2[1]]
    
    return go.Scatter(x=arrow_x, y=arrow_y, mode='lines',
                     line=dict(color=color, width=2), showlegend=False)

# 시뮬레이션 플롯 생성 함수.    
def create_simulation_plot():
    state = st.session_state.simulation_state
    center = [WINDOW_WIDTH/2, WINDOW_HEIGHT/2]
    
    fig = go.Figure()
    
    # 궤도, 중심점, 공 위치를 한번에 추가
    fig.add_traces([
        # 원형 궤도
        go.Scatter(
            x=center[0] + ORBIT_RADIUS*np.cos(np.linspace(0, 2*np.pi, 100)),
            y=center[1] + ORBIT_RADIUS*np.sin(np.linspace(0, 2*np.pi, 100)),
            mode='lines', line=dict(color='black', width=1)
        ),
        # 중심점
        go.Scatter(
            x=[center[0]], y=[center[1]],
            mode='markers', marker=dict(color='red', size=10)
        ),
        # 공
        go.Scatter(
            x=[state['position'][0]], y=[state['position'][1]],
            mode='markers', marker=dict(color='blue', size=15)
        )
    ])
    
    # 궤적이 있는 경우만 추가
    if state['trajectory_points']:
        trajectory = np.array(state['trajectory_points'])
        fig.add_trace(go.Scatter(
            x=trajectory[:, 0], y=trajectory[:, 1],
            mode='markers', marker=dict(color='black', size=2)
        ))
    
    # 벡터 표시 (속도, 힘)
    for show_vector, vector_type, color in [
        (state['show_velocity_vector'], state['velocity'], 'blue'),
        (state['show_force_vector'], calculate_force(state['position'], state['velocity']), 'green')
    ]:
        if show_vector:
            vector_end = [
                state['position'][0] + vector_type[0] * VECTOR_SCALE,
                state['position'][1] + vector_type[1] * VECTOR_SCALE
            ]
            arrow = create_arrow(state['position'], vector_end, color)
            if arrow:
                fig.add_trace(arrow)
    
    # 레이아웃 설정
    fig.update_layout(
        width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
        showlegend=False,
        xaxis=dict(range=[0, WINDOW_WIDTH], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, WINDOW_HEIGHT]),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white'
    )
    
    return fig

def main():
    init_simulation_state()
    
    # 컨트롤 버튼들을 딕셔너리로 관리
    controls = {
        "▶️ 시뮬레이션 시작/정지": lambda: {'simulation_running': not st.session_state.simulation_state['simulation_running']},
        "🟢 힘 벡터 표시": lambda: {'show_force_vector': not st.session_state.simulation_state['show_force_vector']},
        "🔵 속도 벡터 표시": lambda: {'show_velocity_vector': not st.session_state.simulation_state['show_velocity_vector']},
        "🔄 다시 재생": lambda: {
            'position': [center_x + ORBIT_RADIUS, center_y],
            'velocity': [0, INITIAL_SPEED],
            'trajectory_points': [],
            'simulation_running': False,
            'show_velocity_vector': True,
            'show_force_vector': False,
        },
    }
    
    # 버튼 생성 및 상태 업데이트
    cols = st.columns(len(controls))
    for i, (label, update_func) in enumerate(controls.items()):
        unique_key = f"sim_button_{label}_{st.session_state.get('button_timestamp', 0)}"
        if cols[i].button(label, key=unique_key, type="secondary", use_container_width=True):
            st.session_state.simulation_state.update(update_func())
    
    # 플롯 placeholder 생성
    plot_container = st.empty()
    
    # 시뮬레이션 실행 및 표시
    while st.session_state.simulation_state['simulation_running']:
        update_position(1/FPS)
        fig = create_simulation_plot()
        plot_container.plotly_chart(fig, use_container_width=False)
        time.sleep(1/FPS)  # FPS에 맞춰 대기
    
    # 시뮬레이션이 멈춰있을 때도 현재 상태 표시
    if not st.session_state.simulation_state['simulation_running']:
        fig = create_simulation_plot()
        plot_container.plotly_chart(fig, use_container_width=False)

if __name__ == "__main__":
    main()