import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="스위칭 손실 시뮬레이터", layout="wide")

st.title("스위칭 소자의 도통 손실과 스위칭 손실 시뮬레이터")

st.write(
    """
    이 프로그램은 도통 손실과 스위칭 손실을 비교하고,
    스위칭 이벤트마다 손실이 계단형으로 누적되는 과정을 보여줍니다.
    스위칭 손실은 스틸체스 적분의 사건 기준 누적 관점과 연결됩니다.
    """
)

st.sidebar.header("입력값 설정")

I = st.sidebar.number_input("전류 I [A]", min_value=0.0, value=8.0, step=0.5)
V = st.sidebar.number_input("전압 V [V]", min_value=0.0, value=48.0, step=1.0)
R = st.sidebar.number_input("ON 상태 저항 R [Ω]", min_value=0.0, value=0.025, step=0.005, format="%.5f")
D = st.sidebar.slider("듀티비 D", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

f_sw = st.sidebar.number_input("스위칭 주파수 f_sw [Hz]", min_value=0.1, value=5.0, step=1.0)
t_sw_ns = st.sidebar.number_input("스위칭 시간 t_sw [ns]", min_value=0.0, value=100.0, step=10.0)
t_sw = t_sw_ns * 1e-9

k = st.sidebar.number_input("스위칭 손실 계수 k", min_value=0.0, value=0.5, step=0.1)
error_percent = st.sidebar.slider("스위칭 손실 랜덤 오차 범위 [%]", min_value=0, max_value=100, value=10, step=1)

t_end = st.sidebar.number_input("시뮬레이션 시간 [s]", min_value=0.1, value=5.0, step=0.5)
dt = st.sidebar.number_input("시간 간격 dt [s]", min_value=0.0001, value=0.001, step=0.001, format="%.4f")

random_seed = st.sidebar.number_input("랜덤 시드", min_value=0, value=42, step=1)

# 기본 손실 계산
P_cond = D * (I ** 2) * R
e_sw_base = k * V * I * t_sw

switching_period = 1 / f_sw
time = np.arange(0, t_end + dt, dt)

N_t = np.floor(time / switching_period).astype(int)
N_total = int(N_t[-1])

np.random.seed(int(random_seed))

error_range = error_percent / 100

if N_total > 0:
    random_errors = np.random.uniform(-error_range, error_range, N_total)
    e_sw_each = e_sw_base * (1 + random_errors)
    e_sw_each = np.maximum(e_sw_each, 0)
    E_sw_cumulative_events = np.cumsum(e_sw_each)
else:
    random_errors = np.array([])
    e_sw_each = np.array([])
    E_sw_cumulative_events = np.array([])

E_sw = np.zeros_like(time)

for i in range(len(time)):
    n = N_t[i]
    if n > 0:
        E_sw[i] = E_sw_cumulative_events[n - 1]
    else:
        E_sw[i] = 0

E_cond = P_cond * time
E_total = E_cond + E_sw

P_sw_avg = E_sw[-1] / t_end if t_end > 0 else 0
P_total_avg = P_cond + P_sw_avg

st.subheader("계산 결과")

col1, col2, col3, col4 = st.columns(4)

col1.metric("도통 손실 전력", f"{P_cond:.6f} W")
col2.metric("기준 1회 스위칭 손실", f"{e_sw_base:.10f} J")
col3.metric("평균 스위칭 손실 전력", f"{P_sw_avg:.6f} W")
col4.metric("전체 평균 손실 전력", f"{P_total_avg:.6f} W")

col5, col6, col7 = st.columns(3)

col5.metric("총 스위칭 횟수", f"{N_total} 회")
col6.metric("총 도통 손실 에너지", f"{E_cond[-1]:.6f} J")
col7.metric("총 스위칭 손실 에너지", f"{E_sw[-1]:.6f} J")

st.divider()

st.subheader("1. 각 스위칭 사건의 손실 에너지")

if N_total > 0:
    event_numbers = np.arange(1, N_total + 1)

    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(event_numbers, e_sw_each, marker="o")
    ax1.axhline(e_sw_base, linestyle="--", label="Base switching loss")
    ax1.set_xlabel("Switching Event Number")
    ax1.set_ylabel("Switching Loss per Event [J]")
    ax1.set_title("Random Switching Loss per Event")
    ax1.grid(alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)
else:
    st.warning("시뮬레이션 시간 동안 스위칭 이벤트가 없습니다. f_sw 또는 t_end를 키워보세요.")

st.subheader("2. 누적 손실 에너지 비교")

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
ax2.plot(time, E_cond, label="Conduction Loss Energy")
ax2.step(time, E_sw, where="post", label="Switching Loss Energy")
ax2.plot(time, E_total, label="Total Loss Energy")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Energy [J]")
ax2.set_title("Cumulative Loss Energy")
ax2.grid(alpha=0.3)
ax2.legend()
st.pyplot(fig2)

st.subheader("3. 스위칭 손실만 확대")

fig3, ax3 = plt.subplots(figsize=(9, 4.5))
ax3.step(time, E_sw, where="post")
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Switching Loss Energy [J]")
ax3.set_title("Stepwise Cumulative Switching Loss")
ax3.grid(alpha=0.3)
st.pyplot(fig3)

st.subheader("4. 평균 손실 전력 비교")

loss_names = ["Conduction", "Switching", "Total"]
loss_values = [P_cond, P_sw_avg, P_total_avg]

fig4, ax4 = plt.subplots(figsize=(7, 4.5))
ax4.bar(loss_names, loss_values)
ax4.set_ylabel("Power Loss [W]")
ax4.set_title("Average Power Loss Comparison")
ax4.grid(axis="y", alpha=0.3)
st.pyplot(fig4)

st.divider()

st.subheader("해석")

st.write(
    """
    도통 손실은 전류가 흐르는 시간 동안 계속 발생하므로 누적 그래프가 연속적으로 증가합니다.
    반면 스위칭 손실은 스위칭 이벤트가 발생할 때마다 추가되므로 계단형으로 증가합니다.

    스위칭 손실을 일정한 값으로 두면 `E_sw(t) = e_sw N(t)`로 단순화할 수 있습니다.
    하지만 이 프로그램은 각 스위칭 사건마다 랜덤 오차를 부여하여
    `E_sw(t) = Σ e_sw,i` 형태로 계산합니다.
    이는 스틸체스 적분 `E_sw(t) = ∫ e_sw(t) dN(t)`를
    사건별 누적합으로 구현한 것입니다.
    """
)
