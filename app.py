# 저에너지 주택의 다음 10분 가전 전력 사용량 예측 - Streamlit 시연 앱
# 실행: streamlit run app.py
# 노트북(appliance_energy_project.ipynb)과 같은 전처리·같은 시간순 분할·같은 저장 모델을 사용합니다.

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# app.py가 있는 폴더 기준으로 경로를 잡아야 클라우드에서도 파일을 찾을 수 있음
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'energydata_complete.csv')
# 노트북 4-9에서 joblib으로 저장한 최종 모델 (선형회귀)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'energy_model.pkl')

TARGET = 'Appliances'
# 노트북과 같은 순서여야 모델이 올바르게 예측함
FEATURES = ['hour_sin', 'hour_cos', 'weekday', 'is_weekend', 'lag_10min', 'mean_1h']


def add_time_features(df):
    # 10분 단위이므로 '시 + 분/60'으로 하루 중 위치를 소수로 표현한 뒤
    # 시계 바늘 위치처럼 하루(24시간)를 한 바퀴로 보고 좌표로 변환 (23시 50분과 0시가 가깝도록)
    tod = df['date'].dt.hour + df['date'].dt.minute / 60
    df['hour_sin'] = np.sin(2 * np.pi * tod / 24)
    df['hour_cos'] = np.cos(2 * np.pi * tod / 24)
    df['weekday'] = df['date'].dt.day_of_week          # 월=0 ... 일=6
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    return df


# @st.cache_data: 화면을 조작할 때마다 다시 읽지 않도록 결과를 기억해 둠
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values('date').reset_index(drop=True)
    df = add_time_features(df)

    # 직전 사용량 변수: 현재 값이 섞이지 않도록 shift(1) 뒤에 만듦
    df['lag_10min'] = df[TARGET].shift(1)
    df['mean_1h'] = df[TARGET].shift(1).rolling(6).mean()
    return df.dropna(subset=['lag_10min', 'mean_1h']).reset_index(drop=True)


# @st.cache_resource: 모델 파일은 처음 한 번만 불러오고 계속 재사용
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    # 노트북 4-1과 같은 시간순 분할: 뒤 20% 기간을 테스트로 사용
    data = load_data()
    test_df = data.iloc[int(len(data) * 0.8):].copy()
    test_df['pred'] = np.clip(model.predict(test_df[FEATURES]), 0, None)

    y = test_df[TARGET]
    scores = {
        'R²': r2_score(y, test_df['pred']),
        'RMSE': np.sqrt(mean_squared_error(y, test_df['pred'])),
        'MAE': mean_absolute_error(y, test_df['pred']),
        # 비교 기준: 10분 전 값을 그대로 쓰는 베이스라인
        'base_R²': r2_score(y, test_df['lag_10min']),
    }
    return model, scores, test_df


def make_input(when, lag_10min, mean_1h):
    # 화면에서 받은 값을 학습 때와 똑같은 컬럼 형태로 만듦
    df = add_time_features(pd.DataFrame({'date': [pd.Timestamp(when)]}))
    df['lag_10min'] = lag_10min
    df['mean_1h'] = mean_1h
    return df[FEATURES]


# ---------------- 화면 구성 ----------------
st.set_page_config(page_title='가전 전력 사용량 예측', layout='wide')
st.title('저에너지 주택의 다음 10분 가전 전력 사용량 예측')
st.caption('데이터: UCI Appliances Energy Prediction (벨기에 저에너지 주택, 2016-01-11 ~ 2016-05-27, 10분 간격) / 모델: 선형회귀')

for path in [DATA_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        st.error(f'파일이 없습니다: {path}')
        st.stop()

data = load_data()
model, scores, test_df = load_model()

# 모델 성능 (노트북 4-4와 같은 시간순 테스트 기간)
st.markdown(f"**시간순 테스트 기간 성능** ({test_df['date'].min():%Y-%m-%d} ~ {test_df['date'].max():%Y-%m-%d})")
c1, c2, c3, c4 = st.columns(4)
c1.metric('테스트 R²', f"{scores['R²']:.3f}")
c2.metric('테스트 RMSE', f"{scores['RMSE']:.1f} Wh")
c3.metric('테스트 MAE', f"{scores['MAE']:.1f} Wh")
c4.metric('베이스라인 R² (10분 전 값)', f"{scores['base_R²']:.3f}")

tab1, tab2 = st.tabs(['직접 입력해서 예측', '테스트 기간 실제 vs 예측'])

# 1) 직접 입력: 시각 + 직전 사용량 → 다음 10분 예측
with tab1:
    st.sidebar.header('예측 조건 입력')
    day = st.sidebar.date_input('날짜', value=pd.Timestamp('2016-05-16'))
    hour = st.sidebar.slider('시간(시)', 0, 23, 18)
    minute = st.sidebar.select_slider('분', options=[0, 10, 20, 30, 40, 50], value=0)
    # 슬라이더 범위는 실제 데이터의 최솟값~최댓값
    lo, hi = int(data[TARGET].min()), int(data[TARGET].max())
    lag_10min = st.sidebar.slider('10분 전 사용량 (Wh)', lo, hi, 100, step=10)
    mean_1h = st.sidebar.slider('직전 1시간 평균 사용량 (Wh)', lo, hi, 100, step=5)

    when = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)
    pred = max(model.predict(make_input(when, lag_10min, mean_1h))[0], 0)   # 사용량은 음수가 될 수 없음
    st.subheader(f'{when:%Y-%m-%d %H:%M} 예상 사용량: 약 {pred:,.0f} Wh (10분간)')
    st.caption(f'요일: {"월화수목금토일"[when.weekday()]}요일 / 10분 전 {lag_10min} Wh, 직전 1시간 평균 {mean_1h} Wh 기준')

    # 같은 직전 사용량이라고 가정하고 시간대만 바꿨을 때의 예측 (시간대 효과 확인용)
    times = [pd.Timestamp(day) + pd.Timedelta(hours=h) for h in range(24)]
    preds_24 = [max(model.predict(make_input(t, lag_10min, mean_1h))[0], 0) for t in times]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(range(24), preds_24, marker='o')
    ax.axvline(hour + minute / 60, color='red', linestyle='--', alpha=0.5)
    ax.set_xticks(range(24))
    ax.set_xlabel('Hour')
    ax.set_ylabel('Predicted Wh (10 min)')
    ax.set_title('Prediction by hour (same recent usage)')
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption('빨간 점선은 선택한 시각입니다. 직전 사용량을 고정하고 시간대만 바꾼 참고용 그래프입니다.')

# 2) 테스트 기간: 모델이 학습하지 않은 기간에서 실제값과 예측값 비교
with tab2:
    days = sorted(test_df['date'].dt.date.unique())
    # 첫날(04-30)은 08:30부터라 하루 전체가 아니므로 둘째 날을 기본값으로 둠
    pick = st.selectbox('날짜 선택 (테스트 기간)', days, index=1)
    one = test_df[test_df['date'].dt.date == pick]

    fig2, ax2 = plt.subplots(figsize=(11, 3.8))
    ax2.plot(one['date'], one[TARGET], label='Actual', linewidth=1.2)
    ax2.plot(one['date'], one['pred'], label='Predicted', linewidth=1.2)
    ax2.set_ylabel('Wh (10 min)')
    ax2.set_title(f'{pick}: actual vs predicted (10-minute)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.autofmt_xdate()
    st.pyplot(fig2)

    d1, d2, d3 = st.columns(3)
    d1.metric('이날 실제 합계', f"{one[TARGET].sum() / 1000:.2f} kWh")
    d2.metric('이날 예측 합계', f"{one['pred'].sum() / 1000:.2f} kWh")
    d3.metric('이날 MAE', f"{mean_absolute_error(one[TARGET], one['pred']):.1f} Wh")
    st.caption('예측이 실제보다 10분 늦게 따라가는 경향이 있습니다. 직전 사용량에 크게 의존하는 모델이라 갑자기 튀는 순간은 미리 맞히지 못합니다.')
