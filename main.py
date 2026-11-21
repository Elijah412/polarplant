#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""극지식물 개체별 생육 차이 분석 대시보드 (Streamlit) - 학교명, 열 이름, 시간 형식 무관"""

import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 0. 공통 설정 ===
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")

st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
    html, body, [class*="css"] {
       font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True)

# === 1. CSV 읽기 헬퍼 (utf-8-sig -> cp949) ===

def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    # 바이트 버퍼를 받아 utf-8-sig, cp949 순서로 시도
    for enc in ('utf-8-sig', 'cp949'):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


_FILE_CACHE = {}

def read_file(file) -> pd.DataFrame:
    # st.file_uploader 객체를 받아 한 번만 읽기
    # 같은 파일은 캐싱하여 중복 읽기 방지
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    # bytes 해시 키로 캐시 (같은 파일은 한 번만 읽음)
    try:
        key = hash(data)
    except Exception:
        key = id(data)
    if key in _FILE_CACHE:
        return _FILE_CACHE[key]
    result = read_csv_cached(data)
    _FILE_CACHE[key] = result
    return result

# === 2. 열 이름 표준화 ===

def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(' ', '')
        if k in ('time', 'categories', '시간', '날짜', 'datetime', 'date'):
            rename[c] = "time"
        elif k in ('temperature', '온도'):
            rename[c] = "temperature"
        elif k in ('humidity', '습도'):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ('time', '개체번호'):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(',', '').
                str.extract(r'(-?\d+\.?\d*)')[0],
                errors='coerce',
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if '생중량' not in df.columns and {'지상부 질량', '지하부 질량'} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if 'time' in df.columns:
        t = df['time'].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors='coerce', format='mixed')
        bad = parsed.isna() & t.str.match(r'^\d{8}_\d{1,2}$')
        if bad.any():
            parts = t[bad].str.split('_', expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format='%Y%m%d')
                + pd.to_timedelta(parts[1].astype(int), unit='h')
            )
        df['time'] = parsed
    return df

# === 3. 학교명 추출 (파일명에서) ===

def extract_school(filename: str) -> str:
    if not filename:
        return "업로드된 파일"
    base = os.path.basename(filename)
    name = re.sub(r'_[-\s\S]*\.csv$', '', base, flags=re.IGNORECASE)
    name = re.sub(r'-[-\s\S]*$', '', name)
    name = re.sub(r'[^a-zA-Z가-힣0-9\s]', '', name)
    return name.strip() or '우리 학교'

# === 4. 수치 요약 헬퍼 ===

def safe_mean(series) -> float:
    s = pd.to_numeric(series, errors='coerce').dropna()
    return float(s.mean()) if len(s) >= 2 else float('nan')

def safe_min(series) -> float:
    s = pd.to_numeric(series, errors='coerce').dropna()
    return float(s.min()) if len(s) >= 1 else float('nan')

def safe_max(series) -> float:
    s = pd.to_numeric(series, errors='coerce').dropna()
    return float(s.max()) if len(s) >= 1 else float('nan')

def safe_std(series) -> float:
    s = pd.to_numeric(series, errors='coerce').dropna()
    return float(s.std()) if len(s) >= 2 else float('nan')

def safe_corr(x, y) -> float:
    df2 = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'),
                        'y': pd.to_numeric(y, errors='coerce')}).dropna()
    if len(df2) < 2:
        return float('nan')
    return float(df2['x'].corr(df2['y']))

# === 5. 그래프 한글 폰트 ===

def set_korean(fig):
    fig.update_layout(
        font=dict(family='Malgun Gothic, Apple SD Gothic Neo, sans-serif')
    )
    return fig

# === 6. 일별 평균 꺾은선 ===

def daily_line(df, col, title, color):
    _df = df[['time', col]].dropna().copy()
    if len(_df) < 2:
        return None
    _df['day'] = _df['time'].dt.date
    daily = _df.groupby('day')[col].mean().reset_index()
    daily.columns = ['day', col]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['day'], y=daily[col],
        mode='lines', name=title,
        line=dict(color=color, width=2),
        hovertemplate='%s<br>%%{x|%%Y-%%m-%%d} : %%{y:%.1f}<extra></extra>' % title,
    ))
    fig.update_layout(
        title=dict(text=title, x=0.02, y=0.92, xanchor='left'),
        xaxis_title='날짜',
        yaxis_title=title,
        margin=dict(l=50, r=20, t=60, b=40),
        height=220,
        hovermode='x unified',
    )
    return set_korean(fig)

# === 7. 개체별 막대 그래프 (상위 25% 녹색 / 하위 25% 빨강) ===

def bar_individual(df, col, title, color, unit):
    # '개체번호' 열이 반드시 있는지 확인 (없으면 안전하게 None 반환)
    if '개체번호' not in df.columns or col not in df.columns:
        return None
    _df = df[['개체번호', col]].dropna().copy()
    if len(_df) < 2:
        return None
    _df = _df.sort_values(col, ascending=True)
    q1 = _df[col].quantile(0.25)
    q3 = _df[col].quantile(0.75)
    colors = []
    for v in _df[col]:
        if v >= q3:
            colors.append("#2ecc71")   # 상위 25%
        elif v <= q1:
            colors.append("#e74c3c")   # 하위 25%
        else:
            colors.append(color)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=_df['개체번호'], y=_df[col],
        marker=dict(color=colors),
        hovertemplate='%s<br>%%{x} : %%{y:.1f}%s<extra></extra>' % (title, unit),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.02, y=0.92, xanchor='left'),
        xaxis_title='개체번호',
        yaxis_title='%s (%s)' % (title, unit),
        margin=dict(l=50, r=20, t=60, b=40),
        height=320,
        showlegend=False,
    )
    return set_korean(fig)

# === 8. 생중량 분포 박스플롯 ===

def boxplot_weights(df):
    cols_avail = [c for c in ['생중량', '지상부 질량', '지하부 질량'] if c in df.columns]
    if len(cols_avail) < 1 or '개체번호' not in df.columns:
        return None
    _df = df[cols_avail].dropna()
    if len(_df) < 2:
        return None
    fig = go.Figure()
    color_map = {'생중량': '#2980b9', '지상부 질량': '#27ae60', '지하부 질량': '#8e44ad'}
    for c in cols_avail:
        fig.add_trace(go.Box(
            y=_df[c],
            name=c,
            fillcolor=color_map.get(c, '#95a5a6'),
            marker=dict(color=color_map.get(c, '#95a5a6')),
            boxpoints='outliers',
            hovertemplate='%s<br>값 : %%{y:.2f}<extra></extra>' % c,
        ))
    fig.update_layout(
        title=dict(text='생중량 분포 (박스플롯)', x=0.02, y=0.92, xanchor='left'),
        yaxis_title='질량 (g)',
        margin=dict(l=50, r=20, t=60, b=40),
        height=320,
        showlegend=True,
    )
    return set_korean(fig)

# === 9. 산점도 + np.polyfit 추세선 (px.scatter trendline 사용 금지) ===

def scatter_trend(df, x_col, y_col, x_label, y_label):
    _df = df[[x_col, y_col]].dropna().copy()
    if len(_df) < 2:
        return None
    x = _df[x_col].values
    y = _df[y_col].values
    r = safe_corr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = slope * xs + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(color='#2980b9', size=8,
                     line=dict(width=0.5, color='white')),
        hovertemplate='%s : %%{x:.1f}<br>%s : %%{y:.2f}<extra></extra>'
                     % (x_label, y_label),
        name='데이터',
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='lines',
        line=dict(color='#e74c3c', width=2, dash='dot'),
        name='추세선 (r = %.3f)' % r,
    ))
    fig.update_layout(
        title=dict(
            text='%s vs %s (상관계수 r = %.3f)' % (x_label, y_label, r),
            x=0.02, y=0.92, xanchor='left',
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=50, r=20, t=60, b=40),
        height=340,
        showlegend=True,
    )
    return set_korean(fig)

# === 10. CSV 다운로드 버튼 ===

def download_button(label, df, filename):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime='text/csv',
    )

# === 메인 앱 ===

def main():
    st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

    # --- 사이드바 ---
    st.sidebar.header("📁 데이터 업로드")
    env_file = st.sidebar.file_uploader(
        "환경 데이터 CSV", type="csv",
        help="형식 A(time,temperature,humidity,ph,ec) 또는 "
             "형식 B(categories,온도,습도,EC,PH) 모두 지원",
    )
    growth_file = st.sidebar.file_uploader(
        "생육 결과 CSV", type="csv",
        help="개체번호, 잎 수, 지상부 길이, 지하부 길이, 생중량(없으면 질량 합) 포함",
    )
    st.sidebar.markdown('---')
    measure = st.sidebar.selectbox(
        "📋 생육 측정 항목 (그래프 대상)",
        ["생중량", "지상부 질량", "지하부 질량", "잎 수", "지상부 길이", "지하부 길이"],
        index=0,
        help="그래프에 표시할 측정 항목 선택",
    )

    # --- 데이터 로딩 (한 번만) ---
    with st.spinner("📂 CSV 파일을 읽는 중..."):
        raw_env = read_file(env_file)
        raw_growth = read_file(growth_file)
        env_df = std_cols(raw_env) if not raw_env.empty else pd.DataFrame()
        growth_df = std_cols(raw_growth) if not raw_growth.empty else pd.DataFrame()

    school_env = extract_school(env_file.name if env_file else '')
    school_growth = extract_school(growth_file.name if growth_file else '')
    school_name = school_env or school_growth or '우리 학교'
    if env_file or growth_file:
        st.sidebar.success("✅ %s 데이터 로드 완료" % school_name)

    if env_df.empty and growth_df.empty:
        st.info("📎 사이드바에서 환경 CSV와 생육 결과 CSV를 각각 업로드해주세요.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(['📖 실험 개요', '🌡️ 환경 데이터', '📊 생육 결과'])

    # ===== TAB 1 : 실험 개요 =====
    with tab1:
        try:
            st.header("📖 실험 개요")

            with st.expander("🔬 연구 배경 및 목적", expanded=True):
                st.markdown("""**배경**
극지식물은 낮은 온도, 건조한 환경, 강한 자외선 등 혹독한 조건에서
살아남는 식물입니다. 우리 학교에서는 이러한 극지식물을 직접 재배하며
개체마다 어떻게 다르게 자라는지 관찰하고 있습니다.

**목적**
- 개체별 생중량(총 질량) 차이를 평균·표준편차·변동계수(CV)로 파악하기
- 잎 수, 지상부 길이와 생중량 사이에 어떤 관계가 있는지 상관계수로 확인하기
- 생중량이 큰 상위 25% 개체와 작은 하위 25% 개체의 특징을 비교하기

> **용어 설명**
> - **평균(Mean)** : 모든 값을 더해 개수로 나눈 대표값
> - **표준편차** : 값들이 평균에서 얼마나 퍼져 있는지 나타내는 수치
>   (클수록 개체 간 차이가 큼)
> - **변동계수(CV)** : 표준편차를 평균으로 나눈 값.
>   단위가 다른 그룹끼리도 퍼짐 정도를 비교할 수 있음
> - **상관계수(r)** : 두 변수가 함께 움직이는 정도 (-1 ~ +1).
>   +1에 가까울수록 함께 증가, -1에 가까울수록 반대 방향
""")

            st.subheader("📋 데이터 요약")
            summary_rows = []

            if not growth_df.empty and '개체번호' in growth_df.columns:
                n_ind = growth_df['개체번호'].nunique()
                if 'time' in growth_df.columns and growth_df['time'].notna().any():
                    tmin = growth_df['time'].min().strftime('%Y-%m-%d')
                    tmax = growth_df['time'].max().strftime('%Y-%m-%d')
                    period_g = tmin + ' ~ ' + tmax
                else:
                    period_g = '기록 없음'
                avail = [c for c in ['잎 수', '지상부 길이', '지하부 길이', '생중량',
                                    '지상부 질량', '지하부 질량'] if c in growth_df.columns]
                summary_rows.append({
                    "구분": "%s 생육 결과" % school_name,
                    "개체 수": "%d개체" % n_ind,
                    "측정 기간": period_g,
                    "측정 항목": ", ".join(avail) if avail else "없음",
                })

            if not env_df.empty:
                if 'time' in env_df.columns and env_df['time'].notna().any():
                    tmin = env_df['time'].min().strftime('%Y-%m-%d')
                    tmax = env_df['time'].max().strftime('%Y-%m-%d')
                    period_e = tmin + ' ~ ' + tmax
                else:
                    period_e = '기록 없음'
                env_cols = [c for c in ['temperature', 'humidity', 'ph', 'ec']
                            if c in env_df.columns]
                label_map = {'temperature': '온도', 'humidity': '습도',
                             'ph': 'pH', 'ec': 'EC'}
                items = ", ".join(label_map.get(c, c) for c in env_cols) if env_cols else '없음'
                summary_rows.append({
                    "구분": "%s 환경 데이터" % school_name,
                    '개체 수': '%d개 기록' % len(env_df),
                    "측정 기간": period_e,
                    "측정 항목": items,
                })

            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), width='stretch')
            else:
                st.info("📌 표시할 데이터가 없습니다. CSV를 업로드해주세요.")

            st.subheader("📊 주요 지표")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                if not growth_df.empty and '개체번호' in growth_df.columns:
                    n = growth_df['개체번호'].nunique()
                    st.metric(label="총 개체수", value="%d개체" % n)
                else:
                    st.info("📌 총 개체수 : 데이터 없음")

            with c2:
                if not env_df.empty and 'temperature' in env_df.columns:
                    m = safe_mean(env_df['temperature'])
                    if not np.isnan(m):
                        st.metric(label="평균 온도", value="%.1f ℃" % m)
                    else:
                        st.info("📌 평균 온도 : 데이터 부족")
                else:
                    st.info("📌 평균 온도 : 데이터 없음")

            with c3:
                if not env_df.empty and 'humidity' in env_df.columns:
                    m = safe_mean(env_df['humidity'])
                    if not np.isnan(m):
                        st.metric(label="평균 습도", value="%.1f %%" % m)
                    else:
                        st.info("📌 평균 습도 : 데이터 부족")
                else:
                    st.info("📌 평균 습도 : 데이터 없음")

            with c4:
                if not growth_df.empty and '생중량' in growth_df.columns:
                    m = safe_mean(growth_df['생중량'])
                    if not np.isnan(m):
                        st.metric(label="평균 생중량", value="%.2f g" % m)
                    else:
                        st.info("📌 평균 생중량 : 데이터 부족")
                else:
                    st.info("📌 평균 생중량 : 데이터 없음")

        except Exception as e:
            st.warning("⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: %s" % e)

    # ===== TAB 2 : 환경 데이터 =====
    with tab2:
        try:
            if env_df.empty:
                st.info("📎 환경 CSV를 사이드바에서 업로드하면 이 탭의 내용이 표시됩니다.")
            else:
                st.header("🌡️ 환경 데이터")

                st.subheader("📋 환경 요약 (평균 / 최소 / 최대)")
                env_num_cols = [c for c in ['temperature', 'humidity', 'ph', 'ec']
                                if c in env_df.columns]
                if env_num_cols:
                    label_map = {'temperature': '온도 (℃)', 'humidity': '습도 (%)',
                                 'ph': 'pH', 'ec': 'EC (mS/cm)'}
                    rows = []
                    for c in env_num_cols:
                        s = pd.to_numeric(env_df[c], errors='coerce').dropna()
                        if len(s) >= 1:
                            rows.append({
                                "항목": label_map.get(c, c),
                                '평균': round(float(s.mean()), 2),
                                '최소': round(float(s.min()), 2),
                                '최대': round(float(s.max()), 2),
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width='stretch')
                    else:
                        st.info("📌 환경 수치 데이터가 충분하지 않습니다.")
                else:
                    st.info("📌 환경 CSV에 temperature·humidity·ph·ec 열이 없습니다.")

                st.subheader("📈 날짜별 시계열 (일별 평균)")
                t1, t2 = st.columns(2)

                if 'temperature' in env_df.columns:
                    fig = daily_line(env_df, 'temperature',
                                    '온도 변화 (일별 평균)', '#e74c3c')
                    if fig:
                        with t1:
                            st.plotly_chart(fig, width='stretch')
                    else:
                        with t1:
                            st.info("📌 온도 데이터로 일별 그래프를 만들 수 없습니다.")
                else:
                    with t1:
                        st.info("📌 환경 CSV에 temperature 열이 없습니다.")

                if 'humidity' in env_df.columns:
                    fig = daily_line(env_df, 'humidity',
                                    '습도 변화 (일별 평균)', '#3498db')
                    if fig:
                        with t2:
                            st.plotly_chart(fig, width='stretch')
                    else:
                        with t2:
                            st.info("📌 습도 데이터로 일별 그래프를 만들 수 없습니다.")
                else:
                    with t2:
                        st.info("📌 환경 CSV에 humidity 열이 없습니다.")

                if 'ph' in env_df.columns or 'ec' in env_df.columns:
                    t3, t4 = st.columns(2)
                    if 'ph' in env_df.columns:
                        fig = daily_line(env_df, 'ph',
                                          'pH 변화 (일별 평균)', '#9b59b6')
                        if fig:
                            with t3:
                                st.plotly_chart(fig, width='stretch')
                        else:
                            with t3:
                                st.info("📌 pH 데이터로 일별 그래프를 만들 수 없습니다.")
                    else:
                        with t3:
                            st.info("📌 환경 CSV에 ph 열이 없습니다.")
                    if 'ec' in env_df.columns:
                        fig = daily_line(env_df, 'ec',
                                          'EC 변화 (일별 평균)', '#16a085')
                        if fig:
                            with t4:
                                st.plotly_chart(fig, width='stretch')
                        else:
                            with t4:
                                st.info("📌 EC 데이터로 일별 그래프를 만들 수 없습니다.")
                    else:
                        with t4:
                            st.info("📌 환경 CSV에 ec 열이 없습니다.")

                with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                    st.dataframe(env_df, width='stretch')
                    download_button('⬇️ 환경 CSV 다운로드', env_df,
                                    "%s_환경데이터.csv" % school_name)

        except Exception as e:
            st.warning("⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: %s" % e)

    # ===== TAB 3 : 생육 결과 =====
    with tab3:
        try:
            if growth_df.empty:
                st.info("📎 생육 결과 CSV를 사이드바에서 업로드하면 이 탭의 내용이 표시됩니다.")
            else:
                st.header("📊 생육 결과")

                # 생중량 열이 없으면 계산
                if '생중량' not in growth_df.columns:
                    if {'지상부 질량', '지하부 질량'} <= set(growth_df.columns):
                        growth_df["생중량"] = growth_df["지상부 질량"] + growth_df["지하부 질량"]
                        st.info("💡 생중량(g) 열이 없어 '지상부 질량 + 지하부 질량'으로 계산했습니다.")
                    else:
                        st.info("📌 생중량·지상부 질량·지하부 질량 열이 모두 없습니다.")

                st.subheader("🥇 핵심 결과")
                k1, k2, k3 = st.columns(3)

                if '생중량' in growth_df.columns and '개체번호' in growth_df.columns:
                    bw = pd.to_numeric(growth_df['생중량'], errors='coerce').dropna()
                    if len(bw) >= 2:
                        mean_w = float(bw.mean())
                        min_idx = bw.idxmin()
                        max_idx = bw.idxmax()
                        min_val = float(bw.min())
                        max_val = float(bw.max())
                        min_id = str(growth_df.loc[min_idx, '개체번호'])
                        max_id = str(growth_df.loc[max_idx, '개체번호'])
                        with k1:
                            st.metric(label="평균 생중량", value="%.2f g" % mean_w)
                        with k2:
                            st.metric(label="최대 개체", value="%s : %.2f g" % (max_id, max_val))
                        with k3:
                            st.metric(label="최소 개체", value="%s : %.2f g" % (min_id, min_val))
                    else:
                        st.info("📌 생중량 데이터가 충분하지 않습니다.")
                else:
                    st.info("📌 생중량 데이터를 계산할 수 없습니다.")

                st.subheader("📊 개체별 생육 비교")
                bar_specs = [
                    ("생중량",      "평균 생중량",       "#2980b9", "g"),
                    ("잎 수",       "평균 잎 수",        "#27ae60", "장"),
                    ("지상부 길이", "평균 지상부 길이", "#8e44ad", "mm"),
                    ("지하부 길이", "평균 지하부 길이", "#d35400", "mm"),
                ]
                for i in range(0, len(bar_specs), 2):
                    left_pair = bar_specs[i:i+2]
                    col_a, col_b = st.columns(2)
                    for j, (col_name, title, color, unit) in enumerate(left_pair):
                        fig = bar_individual(growth_df, col_name, title, color, unit)
                        target = col_a if j == 0 else col_b
                        with target:
                            if fig:
                                st.plotly_chart(fig, width='stretch')
                            else:
                                st.info("📌 %s 열이 없거나 데이터가 부족합니다." % col_name)

                st.subheader("📦 생중량 분포")
                fig_box = boxplot_weights(growth_df)
                if fig_box:
                    st.plotly_chart(fig_box, width='stretch')
                else:
                    st.info("📌 생중량·지상부 질량·지하부 질량 데이터가 없어 분포를 그릴 수 없습니다.")

                st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")
                scatter_pairs = [
                    ("잎 수",        "생중량",       "잎 수 (장)",        "생중량 (g)"),
                    ("지상부 길이", "생중량",       "지상부 길이 (mm)", "생중량 (g)"),
                ]
                for x_col, y_col, x_lbl, y_lbl in scatter_pairs:
                    if x_col in growth_df.columns and y_col in growth_df.columns:
                        fig_sc = scatter_trend(growth_df, x_col, y_col, x_lbl, y_lbl)
                        if fig_sc:
                            st.plotly_chart(fig_sc, width='stretch')
                        else:
                            st.info("%s·%s 데이터가 충분하지 않습니다." % (x_col, y_col))
                    else:
                        missing = [c for c in [x_col, y_col] if c not in growth_df.columns]
                        st.info("📌 다음 열이 없습니다: %s" % ", ".join(missing))

                with st.expander("📄 생육 데이터 원본 + CSV 다운로드"):
                    st.dataframe(growth_df, width='stretch')
                    download_button('⬇️ 생육 CSV 다운로드', growth_df,
                                    "%s_생육결과.csv" % school_name)

        except Exception as e:
            st.warning("⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: %s" % e)


if __name__ == '__main__':
    main()
