import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import datetime

# 한글 폰트 설정
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False

# 데이터 로드 (캐시)
@st.cache_data
def load_data():
    df = pd.read_excel("앱만들기용 단지데이터 0610버전.xlsx", sheet_name="Sheet1")
    df.set_index("단지 평형", inplace=True)
    return df

df = load_data()
year_cols = [c for c in df.columns if isinstance(c, int)]

# 세션 초기화
st.session_state.setdefault('recent_home', None)
st.session_state.setdefault('recent_move', None)
st.session_state.setdefault('show_home', False)
st.session_state.setdefault('show_move', False)

# CAGR 계산 함수: 첫 데이터 연도와 마지막 데이터 연도의 차이를 사용
def calculate_cagr(series):
    valid = series.dropna()
    if len(valid) < 2:
        return None
    first_year = valid.index[0]
    last_year = valid.index[-1]
    start = valid.iloc[0]
    end = valid.iloc[-1]
    period = last_year - first_year
    if start <= 0 or period <= 0:
        return None
    return (end / start) ** (1/period) - 1

# 예측 가격 생성 함수
def predict_prices(start_price, start_year, cagr, end_year=2032):
    pred = {start_year: start_price}
    if cagr is None:
        return pred
    for y in range(start_year + 1, end_year + 1):
        pred[y] = pred[y-1] * (1 + cagr)
    return pred

# 목표 도달 날짜 계산 함수
def estimate_target_date(start_price, start_year, cagr, goal):
    if goal * 10000 <= start_price:
        return f"{start_year}년 1월 1일"
    if not cagr or cagr <= 0:
        return "도달불가"
    yrs = math.log((goal * 10000) / start_price) / math.log(1 + cagr)
    if yrs > 8:
        return "도달불가"
    days = int(round(yrs * 365.25))
    dt0 = datetime.date(start_year, 1, 1)
    target = dt0 + datetime.timedelta(days=days)
    return target.strftime("%Y년 %m월 %d일")

# 앱 제목
st.title("🏠 GARATA")

# 입력: 내집 / 갈집
col1, col2 = st.columns(2)
with col1:
    st.subheader("내집")
    home_sel = st.selectbox("단지 평형 선택", [""] + list(df.index), key="home")
    신고가_home = st.number_input("신고가 (억)", key="home_price", min_value=0.0, step=0.1, format="%.1f")
    신고년_home = st.selectbox("신고 연도", year_cols, index=year_cols.index(2025), key="home_year")
    목표가_home = st.number_input("목표가 (억)", key="home_goal", min_value=0.0, step=0.1, format="%.1f")
    if st.button("내집 확인"):
        st.session_state.recent_home = home_sel
        st.session_state.show_home = True

with col2:
    st.subheader("갈집")
    move_sel = st.selectbox("단지 평형 선택", [""] + list(df.index), key="move")
    신고가_move = st.number_input("신고가 (억)", key="move_price", min_value=0.0, step=0.1, format="%.1f")
    신고년_move = st.selectbox("신고 연도", year_cols, index=year_cols.index(2025), key="move_year")
    목표가_move = st.number_input("목표가 (억)", key="move_goal", min_value=0.0, step=0.1, format="%.1f")
    if st.button("갈집 확인"):
        st.session_state.recent_move = move_sel
        st.session_state.show_move = True

# 사용법 · 개발자 정보
i1, i2 = st.columns([3, 2])
with i1:
    st.markdown(
        "**📘 사용법**\n\n"
        "1️⃣ 단지를 선택하고 최근 신고가와 목표가를 입력하세요\n\n"
        "2️⃣ ‘내집 확인’ 버튼을 누르면 목표액까지 도달하는 데 걸리는 시간이 계산됩니다\n\n"
        "3️⃣ ‘갈집 확인’ 버튼을 누르면 내집과 갈집의 미래 예상 가격이 계산됩니다"
    )
with i2:
    st.markdown(
        "**👤 개발자**\n\n"
        "압구정 원 부동산중개\n\n"
        "최규호 이사\n\n"
        "📱 010-3065-1780"
    )

# 내집 목표 도달 예상일 표시
if st.session_state.show_home and st.session_state.recent_home:
    home = st.session_state.recent_home
    date_h = estimate_target_date(신고가_home * 10000, 신고년_home,
                                   calculate_cagr(df.loc[home]), 목표가_home)
    st.markdown(f"▶ **내집 목표 도달 예상일:** {date_h}")

# 갈집 목표 도달 예상일 표시
if st.session_state.show_move and st.session_state.recent_move:
    move = st.session_state.recent_move
    date_m = estimate_target_date(신고가_move * 10000, 신고년_move,
                                   calculate_cagr(df.loc[move]), 목표가_move)
    st.markdown(f"▶ **갈집 목표 도달 예상일:** {date_m}")

# 통합 표 & 그래프 (둘 다 눌렀을 때)
if st.session_state.show_home and st.session_state.show_move:
    home = st.session_state.recent_home
    move = st.session_state.recent_move
    pred_h = predict_prices(신고가_home * 10000, 신고년_home,
                              calculate_cagr(df.loc[home]))
    pred_m = predict_prices(신고가_move * 10000, 신고년_move,
                              calculate_cagr(df.loc[move]))

    years = list(range(2025, 2033))
    df_comp = pd.DataFrame({
        "연도": [str(y) for y in years],
        f"{home} (억)": [round(pred_h[y]/10000,1) if y in pred_h else None for y in years],
        f"{move} (억)": [round(pred_m[y]/10000,1) if y in pred_m else None for y in years],
    })
    df_comp["가격차이 (억)"] = df_comp[f"{move} (억)"] - df_comp[f"{home} (억)"]

    # 스타일 적용: 중앙 정렬, 소수점 한자리 포맷, 가격차이 볼드
    styled = (
        df_comp.style
            .format(precision=1)
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")] }])
            .applymap(lambda v: "font-weight: bold" if isinstance(v, (int, float)) else "",
                      subset=["가격차이 (억)"])
    )

    st.dataframe(styled, use_container_width=True)

    # 그래프 출력
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, color in [(home, "#FF2DF1"), (move, "#00CAFF")]:
        vals = df_comp[f"{label} (억)"].astype(float)
        ax.plot(df_comp["연도"], vals, marker='o', label=label, color=color)
        for x, y in zip(df_comp["연도"], vals):
            ax.text(x, y, f"{y:.1f}", ha='center', va='bottom', fontsize=8)

    ax.set_title("2032년까지 예측 가격 추이")
    ax.set_xlabel("연도")
    ax.set_ylabel("가격 (억)")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
