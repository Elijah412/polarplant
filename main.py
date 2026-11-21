import io
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 CSS ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── CSV 읽기 (인코딩 폴리백) ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv_bytes(file) -> pd.DataFrame:
    """file 객체를 utf-8-sig → cp949 순서로 읽고 DataFrame 반환."""
    return _read_csv_impl(file.getvalue())

def _read_csv_impl(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 ──────────────────────────────────────────────
def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
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
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

# ── 학교명 추출 (파일명 앞부분) ─────────────────────────────────
def extract_school(filename: str) -> str:
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    stem = stem.replace("(", "").replace(")", "").strip()
    return stem

# ── Plotly 한글 폰트 세팅 헬퍼 ──────────────────────────────────
def _font(fig):
    fig.update_layout(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
    return fig

# ── 추세선 추가 ──────────────────────────────────────────────────
def add_trendline(fig, x, y, color="red"):
    clean = pd.concat([x, y], axis=1).dropna()
    if len(clean) < 2:
        return
    xi = clean.iloc[:, 0].values
    yi = clean.iloc[:, 1].values
    k, b = np.polyfit(xi, yi, 1)
    fig.add_trace(go.Scatter(
        x=xi, y=k * xi + b,
        mode="lines", name=f"추세선 (기울기 {k:.3f})",
        line=dict(color=color, dash="dot", width=2),
        showlegend=True,
    ))

# ── 막대 그래프 ──────────────────────────────────────────────────
def bar_chart(labels, values, title, color="steelblue"):
    if len(labels) == 0:
        return None
    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        marker_color=color,
        text=values, textposition="outside",
        hoverinfo="x+y",
    )])
    fig.update_layout(
        title=title,
        xaxis_title="개체",
        yaxis_title="값",
        margin=dict(t=40, b=40, l=40, r=20),
        height=360,
        xaxis=dict(tickangle=45, type="category"),
    )
    return _font(fig)

# ── 박스 플롯 ────────────────────────────────────────────────────
def box_chart(values, title):
    if len(values) < 2:
        return None
    fig = go.Figure(data=[go.Box(y=values, boxpoints="outliers", name=title)])
    fig.update_layout(
        title=title,
        yaxis_title="값",
        margin=dict(t=40, b=40, l=40, r=20),
        height=320,
    )
    return _font(fig)

# ── 산점도 (추세선 포함) ─────────────────────────────────────────
def scatter_chart(x, y, xlabel, ylabel, title):
    clean = pd.concat([x, y], axis=1).dropna()
    if len(clean) < 2:
        return None
    xi = clean.iloc[:, 0]
    yi = clean.iloc[:, 1]
    fig = go.Figure(data=[go.Scatter(
        x=xi, y=yi, mode="markers",
        marker=dict(size=8, color="steelblue", opacity=0.7),
        hoverinfo="x+y",
        name="개체",
    )])
    add_trendline(fig, xi, yi)
    r = np.corrcoef(xi, yi)[0, 1]
    fig.update_layout(
        title=f"{title}  (상관계수 r = {r:.3f})",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        margin=dict(t=40, b=40, l=40, r=20),
        height=340,
    )
    return _font(fig)

# ════════════════════════════════════════════════════════════════
#  메인 앱
# ════════════════════════════════════════════════════════════════
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

st.sidebar.header("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.header("📏 생육 측정 항목")
meas_cols = []
if growth_file is not None:
    raw_growth = _read_csv_impl(growth_file.getvalue())
    gdf = std_cols(raw_growth)
    meas_cols = [
        c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
        if c in gdf.columns and pd.api.types.is_numeric_dtype(gdf[c])
    ]
selected_items = st.sidebar.multiselect(
    "비교할 항목",
    options=meas_cols,
    default=meas_cols[:1] if meas_cols else [],
)

school_name = extract_school(
    env_file.name if env_file else (growth_file.name if growth_file else "Unknown")
)
st.sidebar.caption(f"🏫 학교명: {school_name}")

# ── 데이터 로드 (한 번만) ────────────────────────────────────────
env_df = pd.DataFrame()
growth_df = pd.DataFrame()

with st.spinner("환경 데이터를 읽고 있습니다 …"):
    if env_file is not None:
        try:
            env_df = std_cols(_read_csv_impl(env_file.getvalue()))
        except Exception:
            env_df = pd.DataFrame()

with st.spinner("생육 데이터를 읽고 있습니다 …"):
    if growth_file is not None:
        try:
            growth_df = std_cols(growth_file.getvalue())
        except Exception:
            growth_df = pd.DataFrame()

# ── 탭 ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ──────────────────────────────────────────────────────────────────
# Tab 1 : 실험 개요
# ──────────────────────────────────────────────────────────────────
with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.markdown("""
**극지식물(Eskimo mouse-ear chickweed, *Cerastium arcticum*)** 은 낮은 온도와
건조한 환경에서도 생존할 수 있는 식물입니다.
이 대시보드는 우리 학교에서 재배 중인 극지식물 개체들의 **생중량 차이를 비교**하고,
**잎 수·지상부 길이와 생중량의 관계**를 살펴봅니다.

📌 **주요 분석**
- 개체별 생중량 분포 (평균·표준편차·변동계수 CV)
- 잎 수·지상부 길이와 생중량의 상관관계
- 생중량 상위 25% vs 하위 25% 개체 비교
        """.strip())

        st.markdown("---")
        st.header("데이터 요약")

        total_growth = len(growth_df)
        env_metrics = {}

        if not env_df.empty and "time" in env_df.columns:
            valid_t = env_df["time"].dropna()
            env_time_min = ""
            env_time_max = ""
            if len(valid_t) > 0:
                env_time_min = valid_t.min().strftime("%Y-%m-%d %H:%M")
                env_time_max = valid_t.max().strftime("%Y-%m-%d %H:%M")
            for c in ["temperature", "humidity", "ph", "ec"]:
                if c in env_df.columns:
                    s = pd.to_numeric(env_df[c], errors="coerce").dropna()
                    if len(s) > 0:
                        env_metrics[c] = {
                            "mean": round(s.mean(), 2),
                            "min": round(s.min(), 2),
                            "max": round(s.max(), 2),
                        }
        else:
            env_time_min = ""
            env_time_max = ""

        growth_time_min = ""
        growth_time_max = ""
        if not growth_df.empty and "time" in growth_df.columns:
            vt = growth_df["time"].dropna()
            if len(vt) > 0:
                growth_time_min = vt.min().strftime("%Y-%m-%d %H:%M")
                growth_time_max = vt.max().strftime("%Y-%m-%d %H:%M")

        summary_rows = []
        summary_rows.append({
            "항목": "총 개체 수",
            "값": f"{total_growth}개",
            "비고": "생육 CSV 기준",
        })
        if env_time_min:
            summary_rows.append({
                "항목": "환경 측정 기간",
                "값": f"{env_time_min} ~ {env_time_max}",
                "비고": "환경 CSV 기준",
            })
        if growth_time_min:
            summary_rows.append({
                "항목": "생육 측정 기간",
                "값": f"{growth_time_min} ~ {growth_time_max}",
                "비고": "생육 CSV 기준",
            })
        summary_rows.append({
            "항목": "환경 측정 항목",
            "값": ", ".join(sorted(env_metrics.keys())) if env_metrics else "없음",
            "비고": "",
        })
        summary_rows.append({
            "항목": "생육 측정 항목",
            "값": ", ".join(selected_items) if selected_items else "없음",
            "비고": "",
        })

        st.dataframe(pd.DataFrame(summary_rows), width="stretch")

        st.markdown("---")
        st.header("주요 지표")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("총 개체수", f"{total_growth} 개체")

        avg_temp = None
        tmp = pd.to_numeric(env_df["temperature"], errors="coerce").dropna() if "temperature" in env_df.columns else pd.Series(dtype=float)
        if len(tmp) >= 1:
            avg_temp = f"{tmp.mean():.1f} °C"
        col2.metric("평균 온도", avg_temp or "데이터 없음")

        avg_hum = None
        hum = pd.to_numeric(env_df["humidity"], errors="coerce").dropna() if "humidity" in env_df.columns else pd.Series(dtype=float)
        if len(hum) >= 1:
            avg_hum = f"{hum.mean():.1f} %"
        col3.metric("평균 습도", avg_hum or "데이터 없음")

        avg_growth = None
        gr = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna() if "생중량" in growth_df.columns else pd.Series(dtype=float)
        if len(gr) >= 1:
            avg_growth = f"{gr.mean():.2f} g"
        col4.metric("평균 생중량", avg_growth or "데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 실험 개요 표시 중 문제가 생겼어요: {e}")

# ──────────────────────────────────────────────────────────────────
# Tab 2 : 환경 데이터
# ──────────────────────────────────────────────────────────────────
with tab2:
    try:
        if env_df.empty:
            st.info("환경 데이터 CSV를 업로드하면 이 탭의 차트와 요약표가 나타납니다.")
        else:
            st.header("🌡️ 환경 데이터 요약")

            env_lines = []
            for c in ["temperature", "humidity", "ph", "ec"]:
                if c in env_df.columns:
                    s = pd.to_numeric(env_df[c], errors="coerce").dropna()
                    if len(s) > 0:
                        env_lines.append({
                            "측정 항목": c.capitalize(),
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                            "측정 건수": len(s),
                        })
            if env_lines:
                st.dataframe(pd.DataFrame(env_lines), width="stretch")
            else:
                st.info("환경 데이터에 숫자형 측정 열이 없어요.")

            st.markdown("---")
            st.header("날짜별 시계열")

            if "time" in env_df.columns:
                tmp_s = pd.to_numeric(env_df["temperature"], errors="coerce") if "temperature" in env_df.columns else pd.Series(dtype=float)
                hum_s = pd.to_numeric(env_df["humidity"], errors="coerce") if "humidity" in env_df.columns else pd.Series(dtype=float)
                ph_s  = pd.to_numeric(env_df["ph"], errors="coerce") if "ph" in env_df.columns else pd.Series(dtype=float)
                ec_s  = pd.to_numeric(env_df["ec"], errors="coerce") if "ec" in env_df.columns else pd.Series(dtype=float)

                has_temp = tmp_s.notna().sum() > 0
                has_hum  = hum_s.notna().sum() > 0
                has_ph   = ph_s.notna().sum() > 0
                has_ec   = ec_s.notna().sum() > 0

                if not (has_temp or has_hum or has_ph or has_ec):
                    st.info("환경 데이터에서 시계열로 그릴 숫자 값이 없어요.")
                else:
                    env_df2 = env_df[["time"]].copy()
                    if has_temp:
                        env_df2["temperature"] = tmp_s
                    if has_hum:
                        env_df2["humidity"] = hum_s
                    if has_ph:
                        env_df2["ph"] = ph_s
                    if has_ec:
                        env_df2["ec"] = ec_s
                    env_df2 = env_df2.dropna(subset=["time"]).copy()
                    env_df2["date"] = env_df2["time"].dt.date

                    daily_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df2.columns]
                    if daily_cols:
                        daily = env_df2.groupby("date")[daily_cols].mean()

                        subplot_titles = []
                        for c in daily_cols:
                            label_map = {
                                "temperature": "온도 변화 (℃)",
                                "humidity": "습도 변화 (%)",
                                "ph": "pH 변화",
                                "ec": "EC 변화 (μS/cm)",
                            }
                            subplot_titles.append(label_map.get(c, c))

                        fig = make_subplots(
                            rows=len(daily_cols), cols=1,
                            subplot_titles=subplot_titles,
                            vertical_spacing=0.12,
                        )
                        colors = {
                            "temperature": "#e74c3c",
                            "humidity": "#3498db",
                            "ph": "#9b59b6",
                            "ec": "#27ae60",
                        }
                        for i, c in enumerate(daily_cols, 1):
                            fig.add_trace(
                                go.Scatter(
                                    x=daily.index,
                                    y=daily[c],
                                    mode="lines",
                                    name=c,
                                    line=dict(color=colors.get(c, "grey"), width=2),
                                ),
                                row=i, col=1,
                            )
                        fig.update_layout(
                            height=200 * len(daily_cols) + 60,
                            margin=dict(t=40, b=40, l=40, r=20),
                            showlegend=False,
                        )
                        fig.update_xaxes(title_text="날짜", row=len(daily_cols), col=1)
                        fig = _font(fig)
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("시계열로 그릴 환경 측정 항목이 없어요.")
            else:
                st.info("환경 데이터에 'time' 열이 없어요.")

            with st.expander("📋 환경 데이터 원본 테이블 + CSV 다운로드"):
                if env_df.empty:
                    st.info("표시할 환경 데이터가 없어요.")
                else:
                    st.dataframe(env_df, width="stretch")
                    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 환경 데이터 CSV 다운로드 (한글 깨짐 방지)",
                        data=csv_bytes,
                        file_name="환경데이터_다운로드.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 환경 데이터 표시 중 문제가 생겼어요: {e}")

# ──────────────────────────────────────────────────────────────────
# Tab 3 : 생육 결과
# ──────────────────────────────────────────────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("생육 결과 데이터 CSV를 업로드하면 이 탭의 분석과 차트가 나타납니다.")
        else:
            # ── 핵심 결과 카드 ────────────────────────────────────
            st.header("🥇 핵심 결과")

            growth_val = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna() if "생중량" in growth_df.columns else pd.Series(dtype=float)

            if len(growth_val) >= 1:
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("평균 생중량", f"{growth_val.mean():.2f} g")
                cc2.metric(
                    "표준편차 (SD)",
                    f"{growth_val.std():.2f} g",
                    help="표준편차(SD): 값들이 평균에서 얼마나 흩어져 있는지 나타내는 지표예요.",
                )
                cv = (growth_val.std() / growth_val.mean() * 100) if growth_val.mean() != 0 else 0.0
                cc3.metric(
                    "변동계수 (CV)",
                    f"{cv:.1f}%",
                    help="CV(변동계수): 표준편차를 평균으로 나눈 상대 지표예요. 단위가 다른 값도 비교 가능해요.",
                )
                cc4.metric("측정 개체 수", f"{len(growth_val)} 개체")

                max_idx = growth_val.idxmax()
                min_idx = growth_val.idxmin()

                if "개체번호" in growth_df.columns:
                    max_name = str(growth_df.loc[max_idx, "개체번호"])
                    min_name = str(growth_df.loc[min_idx, "개체번호"])
                else:
                    max_name = str(max_idx)
                    min_name = str(min_idx)

                st.markdown(f"**최대 개체:** `{max_name}`  ({growth_val.max():.2f} g)")
                st.markdown(f"**최소 개체:** `{min_name}`  ({growth_val.min():.2f} g)")

                # 상위/하위 25% 설명
                if len(growth_val) >= 4:
                    q75 = growth_val.quantile(0.75)
                    q25 = growth_val.quantile(0.25)
                    n_top = int((growth_val >= q75).sum())
                    n_bot = int((growth_val <= q25).sum())
                    st.markdown(
                        f"**상위 25% 경계값:** {q75:.2f} g  ({n_top}개체)  |  "
                        f"**하위 25% 경계값:** {q25:.2f} g  ({n_bot}개체)"
                    )
            else:
                st.info("생중량 열이 없거나 값이 없어 핵심 결과를 계산할 수 없어요.")

            st.markdown("---")

            # ── 개체별 생육 비교 (막대 그래프) ─────────────────────
            st.header("개체별 생육 비교")
            comp_cols = [
                c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                if c in growth_df.columns and pd.api.types.is_numeric_dtype(growth_df[c])
            ]
            comp_cols = [c for c in comp_cols if c in selected_items] if selected_items else comp_cols

            if not comp_cols:
                st.info("선택한 측정 항목이 없거나, 생육 데이터에 해당 열이 없어요.")
            else:
                for c in comp_cols:
                    vals = pd.to_numeric(growth_df[c], errors="coerce").dropna()
                    if len(vals) < 1:
                        st.info(f"`{c}` 열에 표시할 값이 없어요.")
                        continue
                    idx_list = vals.index.tolist()
                    if "개체번호" in growth_df.columns:
                        labels = [str(growth_df.loc[i, "개체번호"]) for i in idx_list]
                    else:
                        labels = [str(i) for i in idx_list]

                    if c == "생중량" and len(vals) >= 4:
                        q75 = vals.quantile(0.75)
                        q25 = vals.quantile(0.25)
                        bar_colors = []
                        for v in vals:
                            if v >= q75:
                                bar_colors.append("#2ecc71")
                            elif v <= q25:
                                bar_colors.append("#e74c3c")
                            else:
                                bar_colors.append("steelblue")
                    else:
                        bar_colors = "steelblue"

                    bar = bar_chart(labels, vals.tolist(), f"개체별 {c}", color=bar_colors)
                    if bar is not None:
                        st.plotly_chart(bar, width="stretch")

            st.markdown("---")

            # ── 생중량 분포 (박스 플롯) ─────────────────────────────
            if "생중량" in growth_df.columns:
                gv = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna()
                if len(gv) >= 2:
                    bc = box_chart(gv.tolist(), "생중량 분포 (박스 플롯)")
                    if bc is not None:
                        st.plotly_chart(bc, width="stretch")
                else:
                    st.info("생중량 값이 2개 미만이라 박스 플롯을 그릴 수 없어요.")

            st.markdown("---")

            # ── 상관관계 분석 ───────────────────────────────────────
            st.header("상관관계 분석")
            st.markdown("""
상관계수(r)는 **두 변수가 함께 움직이는 정도**를 −1에서 +1 사이로 나타낸 값이에요.
- r이 **+1에 가까울수록** 함께 증가 (양의 상관)
- r이 **−1에 가까울수록** 한쪽이 증가하면 다른 쪽은 감소 (음의 상관)
- r이 **0에 가까울수록** 관계가 거의 없어요
            """.strip())

            corr_pairs = [
                ("잎 수", "생중량"),
                ("지상부 길이", "생중량"),
            ]

            for xcol, ycol in corr_pairs:
                if xcol in growth_df.columns and ycol in growth_df.columns:
                    xv = pd.to_numeric(growth_df[xcol], errors="coerce")
                    yv = pd.to_numeric(growth_df[ycol], errors="coerce")
                    sc = scatter_chart(xv, yv, xcol, ycol, f"{xcol} vs {ycol}")
                    if sc is not None:
                        st.plotly_chart(sc, width="stretch")
                    else:
                        st.info(f"{xcol} vs {ycol} 데이터가 충분하지 않아 상관분석을 건너뛰어요.")
                else:
                    missing = [c for c in [xcol, ycol] if c not in growth_df.columns]
                    st.info(f"다음 열이 생육 데이터에 없어서 {xcol} vs {ycol} 상관분석을 건너뛰어요: {', '.join(missing)}")

            with st.expander("📋 생육 데이터 원본 테이블 + CSV 다운로드"):
                if growth_df.empty:
                    st.info("표시할 생육 데이터가 없어요.")
                else:
                    st.dataframe(growth_df, width="stretch")
                    csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 생육 데이터 CSV 다운로드 (한글 깨짐 방지)",
                        data=csv_bytes,
                        file_name="생육결과_다운로드.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 생육 결과 표시 중 문제가 생겼어요: {e}")
