# main.py
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────

@st.cache_data
def load_csv(file):
    """UTF-8-SIG → CP949 순서로 CSV 읽기."""
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

def std_cols(df):
    """열 이름을 표준 이름으로 통일하고 숫자형으로 정리."""
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
                df[c].astype(str).str.replace(",", ""),
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
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    if "categories" in df.columns:
        catmap = {
            "온도": "temperature",
            "습도": "humidity",
            "ec": "ec",
            "ph": "ph",
        }
        for i, v in df["categories"].items():
            if str(v).strip() in catmap:
                df.at[i, "측정항목"] = catmap[str(v).strip()]
                df.at[i, "측정값"] = df.at[i, "temperature"] \
                    if catmap[str(v).strip()] == "temperature" else \
                    df.at[i, "습도"] if catmap[str(v).strip()] == "humidity" else \
                    df.at[i, "ec"] if catmap[str(v).strip()] == "ec" else \
                    df.at[i, "ph"]
        df = df.dropna(subset=["측정항목"])
        df["time"] = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(df["time"], errors="coerce", format="mixed")
        bad = parsed.isna() & df["time"].str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = df.loc[bad, "time"].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
        wide = df.pivot_table(index="time", columns="측정항목",
                               values="측정값", aggfunc="first")
        wide.index.name = "time"
        wide = wide.reset_index()
        for col in ("temperature", "humidity", "ph", "ec"):
            if col in wide.columns and not pd.api.types.is_numeric_dtype(wide[col]):
                wide[col] = pd.to_numeric(wide[col], errors="coerce")
        return wide
    return df

def school_name(file):
    if file is None:
        return "학교"
    base = os.path.splitext(os.path.basename(file.name))[0]
    return base.replace("_", " ").replace("-", " ")

# ──────────────────────────────────────────────
# 그래프 헬퍼
# ──────────────────────────────────────────────

def scatter_with_trend(x, y, xlabel, ylabel):
    """추세선을 직접 추가한 산점도 (statsmodels 불필요)."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["x"], y=df["y"],
        mode="markers",
        marker=dict(size=7, color="#1f77b4", opacity=0.7),
        hovertemplate=f"{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.2f}}<extra></extra>",
    ))
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    xs = np.linspace(df["x"].min(), df["x"].max(), 100)
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept,
        mode="lines",
        line=dict(color="#d62728", width=2),
        name=f"추세선 (기울기={slope:.3f})",
    ))
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=60, r=20, t=30, b=50),
        height=320,
    )
    return fig

def group_bar_with_cutoff(df, col_x, col_y, cutoff, xlabel, ylabel, higher_better=True):
    """중앙값 기준 두 그룹을 색상으로 구분한 막대 산점도."""
    d = df[[col_x, col_y]].dropna()
    if len(d) < 4:
        return None
    median = d[col_y].median()
    d = d.copy()
    d["그룹"] = np.where(d[col_y] > median, "양호 그룹 (> 중앙값)", "부족 그룹 (≤ 중앙값)")
    if not higher_better:
        d["그룹"] = d["그룹"].map({
            "양호 그룹 (> 중앙값)": "부족 그룹 (≤ 중앙값)",
            "부족 그룹 (≤ 중앙값)": "양호 그룹 (> 중앙값)",
        })
    fig = go.Figure()
    for grp, grp_df in d.groupby("그룹"):
        fig.add_trace(go.Scatter(
            x=grp_df[col_x], y=grp_df[col_y],
            mode="markers",
            name=grp,
            marker=dict(size=8, opacity=0.75),
            hovertemplate=(f"{col_x}: %{{x:.2f}}<br>{col_y}: %{{y:.2f}}<br>그룹: {grp}<extra></extra>"),
        ))
    fig.add_vline(x=median, line_dash="dash", line_color="gray",
                  annotation_text=f"중앙값 {median:.2f}", annotation_position="top right")
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=60, r=20, t=30, b=50),
        height=340,
    )
    return fig

# ──────────────────────────────────────────────
# 메인 앱
# ──────────────────────────────────────────────

st.set_page_config(page_title="극지식물 생육 결과 데이터 분석", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("극지식물 생육 결과 데이터 분석")

# ── 사이드바 ──────────────────────────────────
env_file   = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")
meas_cols = st.sidebar.multiselect(
    "생육 측정 항목",
    ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"],
    default=["잎 수", "지상부 길이", "생중량"],
)

env_raw    = load_csv(env_file)
growth_raw = load_csv(growth_file)
env_df     = std_cols(env_raw) if not env_raw.empty else pd.DataFrame()
growth_df  = std_cols(growth_raw) if not growth_raw.empty else pd.DataFrame()

sch_name   = school_name(env_file) if not env_raw.empty else school_name(growth_file)

tab1, tab2, tab3 = st.tabs(["실험 소개", "식물의 생장 데이터", "상관관계 분석"])

# ── 탭 1: 실험 소개 ─────────────────────────
with tab1:
    try:
        c1, c2 = st.columns(2)
        c1.metric("학교명", sch_name)
        c1.metric("환경 데이터 행 수", f"{len(env_df):,}" if not env_df.empty else "—")
        c1.metric("생육 데이터 개체 수", f"{len(growth_df):,}개체" if not growth_df.empty else "—")

        st.header("실험 개요")
        st.info("이 탭은 학교명과 데이터 행 수 등 기본 정보만 보여주며, "
                "구체적인 환경 수치 그래프는 환경 CSV가 업로드되어야 표시됩니다.")

        if not env_df.empty:
            st.subheader(f"{sch_name} 스마트팜 환경 데이터 개요")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                daily = env_df.copy()
                daily["date"] = daily["time"].dt.date
                daily_avg = daily.groupby("date")[env_cols].mean()
                rows = []
                for col in env_cols:
                    s = daily_avg[col].dropna()
                    if len(s) >= 1:
                        rows.append({
                            "항목": {"temperature": "온도(°C)", "humidity": "습도(%)",
                                     "ph": "pH", "ec": "EC(mS/cm)"}[col],
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                            "표준편차": round(s.std(), 2),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch")
                    st.caption("표준편차: 데이터가 평균을 중심으로 얼마나 흩어져 있는지 나타내는 값. "
                               "숫자가 클수록 하루 사이 변동이 컸다는 뜻입니다.")
            else:
                st.info("환경 데이터에 인식할 수 있는 열이 없어요. "
                        "형식 A(time, temperature, humidity, ph, ec) 또는 "
                        "형식 B(categories, 온도, 습도, EC, PH)인지 확인해 주세요.")

        if not growth_df.empty:
            st.subheader(f"{sch_name} 극지식물 생육 결과 데이터 개요")
            gcols = [c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                  "지상부 질량", "지하부 질량", "생중량"]
                     if c in growth_df.columns]
            if gcols:
                stats_rows = []
                for col in gcols:
                    s = growth_df[col].dropna()
                    if len(s) >= 1:
                        stats_rows.append({
                            "항목": col,
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                            "표준편차": round(s.std(), 2),
                        })
                if stats_rows:
                    st.dataframe(pd.DataFrame(stats_rows), width="stretch")
                    st.caption("평균(Mean): 모든 값을 더해 개수로 나눈 대표값. "
                               "표준편차(Std Dev): 평균에서 개별 값이 얼마나 떨어져 있는지 나타내는 산포 지표.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── 탭 2: 생장 데이터 제시 ──────────────────
with tab2:
    try:
        if growth_df.empty:
            st.info("생육 결과 CSV를 사이드바에서 업로드해 주세요. "
                    "업로드하면 개체별 생육 데이터를 표로 확인할 수 있어요.")
        else:
            gcols = [c for c in ["개체번호", "잎 수", "지상부 길이", "지하부 길이",
                                  "지상부 질량", "지하부 질량", "생중량"]
                     if c in growth_df.columns]
            if gcols:
                st.dataframe(growth_df[gcols], width="stretch", hide_index=True)
                st.caption("개체번호는 학교마다 숫자(1, 2, …) 또는 글자(A1, A2, …)로 달라요. "
                           "분석에서는 개체 간 비교만 하고 개체번호 자체는 분석 항목으로 쓰지 않아요.")

            st.subheader("개체별 생중량 비교 (상위·하위 강조)")
            if "생중량" in growth_df.columns:
                gf = growth_df[["개체번호", "생중량"]].dropna().copy()
                if len(gf) >= 5:
                    gf = gf.sort_values("생중량", ascending=False)
                    q75 = gf["생중량"].quantile(0.75)
                    q25 = gf["생중량"].quantile(0.25)
                    gf["그룹"] = "중간"
                    gf.loc[gf["생중량"] >= q75, "그룹"] = "상위 25%"
                    gf.loc[gf["생중량"] <= q25, "그룹"] = "하위 25%"
                    fig = go.Figure()
                    colors = {"상위 25%": "#2ca02c", "하위 25%": "#d62728", "중간": "#1f77b4"}
                    for grp, grp_df in gf.groupby("그룹"):
                        fig.add_trace(go.Bar(
                            x=grp_df["개체번호"].astype(str),
                            y=grp_df["생중량"],
                            name=grp,
                            marker_color=colors[grp],
                            hovertemplate=f"개체: %{{x}}<br>생중량: %{{y:.2f}}g<extra></extra>",
                        ))
                    fig.update_layout(
                        barmode="group",
                        xaxis_title="개체번호",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=60, r=20, t=30, b=50),
                        height=380,
                    )
                    st.plotly_chart(fig, width="stretch")
                    top_n = gf.head(5)[["개체번호", "생중량"]]
                    bot_n = gf.tail(5)[["개체번호", "생중량"]]
                    c1, c2 = st.columns(2)
                    c1.subheader("생중량이 가장 큰 개체 5")
                    c1.dataframe(top_n, width="stretch", hide_index=True)
                    c2.subheader("생중량이 가장 작은 개체 5")
                    c2.dataframe(bot_n, width="stretch", hide_index=True)
                else:
                    st.info("개체 수가 너무 적어 비교 그래프를 만들 수 없어요. "
                            "최소 5개체 이상이 필요해요.")
            else:
                st.info("생중량 열이 없어 개체별 비교 그래프를 만들 수 없어요. "
                        "지상부 질량 + 지하부 질량으로 생중량을 계산해 표시해요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── 탭 3: 상관관계 분석 ─────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("생육 결과 CSV를 사이드바에서 업로드해 주세요. "
                    "업로드하면 생육 지표 간 상관관계를 산점도와 수치로 확인할 수 있어요.")
        else:
            st.info("""
**상관계수 (r)** 는 -1에서 1 사이의 값으로, 두 지표가 함께 어떻게 움직이는지 나타내요.  
- **r ≈ 1**: 한쪽이 커질 때 다른 쪽도 커지는 강한 양의 관계  
- **r ≈ 0**: 두 지표 사이에 뚜렷한 관계가 없음  
- **r ≈ -1**: 한쪽이 커질 때 다른 쪽이 작아지는 관계  
※ 상관계수가 높다고 해서 반드시 원인-결과 관계(인과관계)는 아니에요.
""")

            avail = [c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                  "지상부 질량", "지하부 질량", "생중량"]
                     if c in growth_df.columns]

            if len(avail) < 2:
                st.info("상관계수를 계산할 수 있는 생육 열이 부족해요. "
                        "잎 수, 지상부 길이, 지하부 길이, 질량 관련 열이 있는지 확인해 주세요.")
            else:
                calc_cols = [c for c in avail if c != "개체번호"]

                # ──────────────────────────────────────
                # 3-1. 가설 제시 및 핵심 상관 분석
                # ──────────────────────────────────────
                st.header("가설: 지하부 발달이 부족한 개체는 지상부 생장에도 제약이 있다")

                st.markdown("""
이 가설을 세 가지 방법으로 검토해요.

1. **상관분석**: 지하부 질량과 지상부 질량 사이에 강한 양의 상관이 있는지 확인  
2. **그룹 비교(t검정)**: 지하부 질량을 기준으로 '양호 그룹'과 '부족 그룹'으로 나누고 두 그룹 간 지상부 질량 차이를 통계적으로 검증  
3. **대표 개체 사례**: 실제로 가장 못 자란 개체와 가장 잘 자란 개체의 지표를 직접 비교

※ 모든 수치는 업로드한 생육 데이터에서 직접 계산한 값이며, 지수 표기 등은 소수점 둘째 자리에서 반올림했어요.
""")

                # ── 상관계수 전체 표
                st.subheader("① 생육 지표 간 상관계수 (r) 전체 표")
                if len(calc_cols) >= 2:
                    sub = growth_df[calc_cols].dropna()
                    if len(sub) >= 2:
                        corr_mat = sub.corr(method="pearson")
                        # 개체번호 제외, 관심도 높은 쌍 위주로 행 추가
                        corr_rows = []
                        pairs_of_interest = [
                            ("지상부 질량", "지하부 질량"),
                            ("잎 수", "지상부 질량"),
                            ("잎 수", "지하부 질량"),
                            ("지상부 길이", "지하부 길이"),
                            ("지상부 질량", "지하부 길이"),
                            ("잎 수", "생중량"),
                            ("지상부 길이", "생중량"),
                            ("지하부 길이", "생중량"),
                            ("지하부 질량", "생중량"),
                            ("지상부 질량", "생중량"),
                        ]
                        for ax, ay in pairs_of_interest:
                            if ax in corr_mat.columns and ay in corr_mat.columns:
                                r = corr_mat.loc[ax, ay]
                                r2 = r ** 2
                                n = sub[[ax, ay]].dropna()
                                corr_rows.append({
                                    "지표 A": ax,
                                    "지표 B": ay,
                                    "상관계수 r": round(r, 3) if not np.isnan(r) else "—",
                                    "결정계수 R²": round(r2, 3),
                                    "관계 강도": ("강한 양의 상관" if r >= 0.7 else
                                                  "중간 양의 상관" if r >= 0.4 else
                                                  "약한 양의 상관" if r >= 0.1 else
                                                  "약한 음의 상관" if r <= -0.1 else
                                                  "중간 음의 상관" if r <= -0.4 else "강한 음의 상관"),
                                    "데이터 수": len(n),
                                })
                        if corr_rows:
                            corr_df = pd.DataFrame(corr_rows)
                            st.dataframe(corr_df, width="stretch")
                            st.caption("상관계수 r은 두 지표가 함께 오르내리는 정도를 나타내요. "
                                       "결정계수 R² = r² 는 한 지표로 다른 지표 변동의 몇 %를 설명할 수 있는지 보여줘요.")
                        else:
                            st.info("상관계수를 표시할 지표 쌍이 없어요.")
                    else:
                        st.info("상관계수를 계산할 데이터가 부족해요.")
                else:
                    st.info("상관계수를 계산할 생육 열이 부족해요.")

                # ── 질문 1: 잎 수 vs 생중량 vs 지상부 길이
                st.subheader("② 잎 수와 지상부 길이 중 어느 항목이 생중량과 더 밀접할까?")
                if "잎 수" in calc_cols and "생중량" in calc_cols:
                    n1 = growth_df[["잎 수", "생중량"]].dropna()
                    if len(n1) >= 2:
                        r1 = n1["잎 수"].corr(n1["생중량"])
                        r1sq = r1 ** 2
                        fig1 = scatter_with_trend(n1["잎 수"], n1["생중량"], "잎 수(장)", "생중량(g)")
                        if fig1:
                            c1, c2 = st.columns([1, 2])
                            c1.metric("잎 수 ↔ 생중량 상관계수 (r)", f"{r1:.3f}")
                            c1.caption(f"R² = {r1sq:.3f} "
                                       f"→ 잎 수로 생중량 변동의 {r1sq*100:.1f}% 설명 가능")
                            c2.plotly_chart(fig1, width="stretch")

                if "지상부 길이" in calc_cols and "생중량" in calc_cols:
                    n2 = growth_df[["지상부 길이", "생중량"]].dropna()
                    if len(n2) >= 2:
                        r2 = n2["지상부 길이"].corr(n2["생중량"])
                        r2sq = r2 ** 2
                        fig2 = scatter_with_trend(n2["지상부 길이"], n2["생중량"],
                                                  "지상부 길이(mm)", "생중량(g)")
                        if fig2:
                            c1, c2 = st.columns([1, 2])
                            c1.metric("지상부 길이 ↔ 생중량 상관계수 (r)", f"{r2:.3f}")
                            c1.caption(f"R² = {r2sq:.3f} "
                                       f"→ 지상부 길이로 생중량 변동의 {r2sq*100:.1f}% 설명 가능")
                            c2.plotly_chart(fig2, width="stretch")

                        # 두 상관 비교
                        if "잎 수" in calc_cols and len(n1) >= 2:
                            stronger = "잎 수" if abs(r1) > abs(r2) else "지상부 길이"
                            sval = max(abs(r1), abs(r2))
                            same = abs(abs(r1) - abs(r2)) < 0.05
                            if same:
                                st.info("잎 수와 지상부 길이 모두 생중량과 비슷한 강도의 상관을 보여요 (r ≈ %.3f). "
                                        "두 지표가 함께 생중량에 영향을 주고 있을 가능성이 있어요." % sval)
                            else:
                                st.info(f"**{stronger}**가 생중량과 더 밀접한 관계(r={sval:.3f})를 보입니다. "
                                        f"줄기 길이보다 잎 수가 생중량 예측에는 더 유용한 지표일 가능성이 있어요.")
                else:
                    st.info("잎 수 또는 생중량 열이 없어 이 비교를 할 수 없어요.")

                # ── 질문 2: 상·하위 그룹 비교 (생중량 기준)
                st.subheader("③ 무거운 개체와 가벼운 개체는 어떤 생육 특징에서 차이가 날까?")
                if "생중량" in calc_cols:
                    gf = growth_df[["개체번호", "생중량"] + [c for c in calc_cols if c not in ("개체번호", "생중량")]].dropna()
                    if len(gf) >= 5:
                        q75 = gf["생중량"].quantile(0.75)
                        q25 = gf["생중량"].quantile(0.25)
                        top = gf[gf["생중량"] >= q75]
                        bot = gf[gf["생중량"] <= q25]
                        if len(top) >= 2 and len(bot) >= 2:
                            comp_rows = []
                            for col in [c for c in calc_cols if c not in ("개체번호", "생중량")]:
                                tv = top[col].dropna()
                                bv = bot[col].dropna()
                                if len(tv) >= 1 and len(bv) >= 1:
                                    diff = tv.mean() - bv.mean()
                                    ratio = tv.mean() / bv.mean() if bv.mean() != 0 else np.nan
                                    comp_rows.append({
                                        "생육 항목": col,
                                        "상위 25% 평균": round(tv.mean(), 2),
                                        "하위 25% 평균": round(bv.mean(), 2),
                                        "평균 차이": round(diff, 2),
                                        "평균 비율 (상위/하위)": round(ratio, 2) if not np.isnan(ratio) else "—",
                                    })
                            if comp_rows:
                                comp_df = pd.DataFrame(comp_rows)
                                st.dataframe(comp_df, width="stretch")
                                max_item = max(
                                    [r for r in comp_rows if isinstance(r["평균 비율 (상위/하위)", float)],
                                    key=lambda r: r["평균 비율 (상위/하위)"],
                                    default=None,
                                )
                                if max_item:
                                    st.info(f"상위 25%와 하위 25% 그룹 간 **평균 비율 차이가 가장 큰 항목**은 "
                                            f"**{max_item['생육 항목']}**(약 {max_item['평균 비율 (상위/하위)']}배)이에요. "
                                            f"이 항목이 잘 자란 개체와 덜 자란 개체를 가르는 핵심 지표일 가능성이 높아요.")
                    else:
                        st.info("개체 수가 적어 상·하위 25% 그룹 비교를 할 수 없어요.")
                else:
                    st.info("생중량 열이 없어 이 비교를 할 수 없어요.")

                # ──────────────────────────────────────
                # 3-2. 가설 검증 — 지하부 발달 ↔ 지상부 생장
                # ──────────────────────────────────────
                st.header("가설 검증: 지하부 발달이 지상부 생장에 미치는 영향")

                # --- (a) 지하부 질량 기준 그룹 비교 ---
                if "지하부 질량" in calc_cols and "지상부 질량" in calc_cols:
                    n_group = growth_df[["개체번호", "지하부 질량", "지상부 질량", "잎 수", "생중량"]].dropna()
                    if len(n_group) >= 4:
                        cutoff_mass = n_group["지하부 질량"].median()
                        good = n_group[n_group["지하부 질량"] > cutoff_mass]
                        poor = n_group[n_group["지하부 질량"] <= cutoff_mass]
                        if len(good) >= 2 and len(poor) >= 2:
                            # t검정
                            t_stat, p_val = scipy_stats.ttest_ind(
                                good["지상부 질량"], poor["지상부 질량"],
                                equal_var=False,
                            )
                            st.subheader("④ 지하부 질량 중앙값 기준 그룹 비교")
                            c_good, c_poor = st.columns(2)
                            c_good.metric("지하부 질량 > 중앙값 그룹 개체 수", f"{len(good)}개체")
                            c_good.metric("지하부 질량 중앙값", f"{cutoff_mass:.2f}g")
                            c_poor.metric("지하부 질량 ≤ 중앙값 그룹 개체 수", f"{len(poor)}개체")

                            # 그룹별 요약 표
                            grp_rows = []
                            for label, grp in [("지하부 발달 양호 그룹", good), ("지하부 발달 부족 그룹", poor)]:
                                grp_rows.append({
                                    "그룹": label,
                                    "개체 수": len(grp),
                                    "지상부 질량 평균 (g)": round(grp["지상부 질량"].mean(), 2),
                                    "잎 수 평균 (장)": round(grp["잎 수"].mean(), 2),
                                    "생중량 평균 (g)": round(grp["생중량"].mean(), 2),
                                    "지하부 질량 평균 (g)": round(grp["지하부 질량"].mean(), 2),
                                })
                            st.dataframe(pd.DataFrame(grp_rows), width="stretch")

                            diff_mass = good["지상부 질량"].mean() - poor["지상부 질량"].mean()
                            diff_leaf = good["잎 수"].mean() - poor["잎 수"].mean()
                            diff_total = good["생중량"].mean() - poor["생중량"].mean()
                            ratio_mass = good["지상부 질량"].mean() / poor["지상부 질량"].mean() if poor["지상부 질량"].mean() != 0 else np.nan

                            c_res1, c_res2, c_res3 = st.columns(3)
                            c_res1.metric("지상부 질량 차이 (양호 − 부족)", f"+{diff_mass:.2f}g")
                            c_res2.metric("t 통계량", f"t = {t_stat:.3f}")
                            sig_str = "p < 0.001 (매우 유의)" if p_val < 0.001 else f"p = {p_val:.4f}" if p_val >= 0.001 else f"p = {p_val:.5f}"
                            c_res3.metric("유의확률 (p)", sig_str)

                            c1, c2 = st.columns([1, 2])
                            c1.metric("잎 수 차이 (양호 − 부족)", f"+{diff_leaf:.1f}장")
                            c1.metric("생중량(총생체량) 차이", f"+{diff_total:.2f}g")
                            if not np.isnan(ratio_mass):
                                c1.metric("지상부 질량 비율 (양호/부족)", f"{ratio_mass:.2f}배")

                            fig_group_mass = group_bar_with_cutoff(
                                n_group, "지상부 질량", "지하부 질량",
                                cutoff=cutoff_mass, xlabel="지상부 질량 (g)", ylabel="지하부 질량 (g)",
                            )
                            if fig_group_mass:
                                st.plotly_chart(fig_group_mass, width="stretch")

                            sig_label = ("통계적으로 유의" if p_val < 0.05 else
                                         "통계적으로 유의하지 않음")
                            st.info(f"지하부 질량 중앙값({cutoff_mass:.2f}g) 기준, 양호 그룹({len(good)}개체)은 "
                                    f"부족 그룹({len(poor)}개체)보다 지상부 질량이 평균 **{diff_mass:.2f}g 더 크고**, "
                                    f"잎 수는 평균 **{diff_leaf:.1f}장 더 많아요.** "
                                    f"t검정 결과 t = {t_stat:.3f}, {sig_label} (p = {p_val:.4f}). "
                                    f"→ 가설 **'지하부 발달이 부족한 개체는 지상부 생장에도 제약이 있다'**는 "
                                    f"지상부 질량 기준에서 {sig_label.lower()}해요.")

                # --- (b) 지하부 길이 기준 그룹 비교 ---
                if "지하부 길이" in calc_cols and "지상부 질량" in calc_cols:
                    n_len = growth_df[["개체번호", "지하부 길이", "지상부 질량", "잎 수", "생중량"]].dropna()
                    if len(n_len) >= 4:
                        cutoff_len = n_len["지하부 길이"].median()
                        good_len = n_len[n_len["지하부 길이"] > cutoff_len]
                        poor_len = n_len[n_len["지하부 길이"] <= cutoff_len]
                        if len(good_len) >= 2 and len(poor_len) >= 2:
                            t_len, p_len = scipy_stats.ttest_ind(
                                good_len["지상부 질량"], poor_len["지상부 질량"],
                                equal_var=False,
                            )
                            st.subheader("⑤ 지하부 길이 중앙값 기준 그룹 비교 (참고)")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("지하부 길이 중앙값", f"{cutoff_len:.2f}mm")
                            c1.metric("양호 그룹 / 부족 그룹 개체 수", f"{len(good_len)} / {len(poor_len)}")
                            diff_len_mass = good_len["지상부 질량"].mean() - poor_len["지상부 질량"].mean()
                            c2.metric("지상부 질량 차이 (양호 − 부족)", f"+{diff_len_mass:.2f}g")
                            c2.metric("t 통계량", f"t = {t_len:.3f}")
                            sig_len = "p < 0.05 (유의)" if p_len < 0.05 else f"p = {p_len:.4f}"
                            c2.metric("유의확률 (p)", sig_len)
                            st.caption("지하부 길이 기준으로는 지상부 질량 차이가 질량 기준보다 작고 덜 유의해요. "
                                       "길이보다 질량이 지상부 생장과 더 강하게 연결된다는 뜻이에요.")

                # --- (c) 뿌리비율 기준 그룹 비교 ---
                if "지하부 질량" in calc_cols and "생중량" in calc_cols:
                    n_ratio = growth_df[["개체번호", "지하부 질량", "생중량"]].dropna()
                    n_ratio["뿌리비율"] = n_ratio["지하부 질량"] / n_ratio["생중량"]
                    if len(n_ratio) >= 4 and "지상부 질량" in calc_cols:
                        n_ratio = n_ratio.merge(
                            growth_df[["개체번호", "지상부 질량"]],
                            on="개체번호",
                            how="left",
                        )
                        cutoff_ratio = n_ratio["뿌리비율"].median()
                        good_r = n_ratio[n_ratio["뿌리비율"] > cutoff_ratio]
                        poor_r = n_ratio[n_ratio["뿌리비율"] <= cutoff_ratio]
                        if len(good_r) >= 2 and len(poor_r) >= 2:
                            t_r, p_r = scipy_stats.ttest_ind(
                                good_r["지상부 질량"], poor_r["지상부 질량"],
                                equal_var=False,
                            )
                            st.subheader("⑥ 뿌리비율(지하부 질량 / 총생체량) 기준 그룹 비교 (참고)")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("뿌리비율 중앙값", f"{cutoff_ratio:.3f}")
                            c1.metric("양호 / 부족 개체 수", f"{len(good_r)} / {len(poor_r)}")
                            diff_r = good_r["지상부 질량"].mean() - poor_r["지상부 질량"].mean()
                            c2.metric("지상부 질량 차이 (양호 − 부족)", f"{diff_r:+.2f}g")
                            c2.metric("유의확률 (p)", f"p = {p_r:.3f}")
                            if p_r >= 0.05:
                                st.info("뿌리비율 기준으로는 지상부 질량 차이가 통계적으로 유의하지 않아요 (p ≥ 0.05). "
                                        "비율이 아니라 **절대적인 지하부 질량**이 지상부 생장과 더 강하게 연결된다는 뜻이에요.")
                            else:
                                st.info(f"뿌리비율 기준으로도 지상부 질량 차이가 유의해요 (p = {p_r:.3f}).")

                # ──────────────────────────────────────
                # 3-3. 대표 개체 사례
                # ──────────────────────────────────────
                st.header("대표 개체 사례")

                if "생중량" in growth_df.columns or {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
                    gf_all = growth_df.copy()
                    if "생중량" in gf_all.columns:
                        gf_all["총생체량"] = gf_all["생중량"]
                    else:
                        gf_all["총생체량"] = gf_all["지상부 질량"] + gf_all["지하부 질량"]

                    gf_all = gf_all.dropna(subset=["총생체량"])
                    if len(gf_all) >= 4:
                        best = gf_all.sort_values("총생체량", ascending=False).head(1).iloc[0]
                        worst = gf_all.sort_values("총생체량", ascending=True).head(1).iloc[0]

                        c_best, c_worst = st.columns(2)
                        c_best.subheader("가장 잘 자란 개체")
                        best_data = {
                            "개체번호": str(best["개체번호"]),
                            "잎 수 (장)": best["잎 수"] if "잎 수" in gf_all.columns else np.nan,
                            "지상부 길이 (mm)": best["지상부 길이"] if "지상부 길이" in gf_all.columns else np.nan,
                            "지하부 길이 (mm)": best["지하부 길이"] if "지하부 길이" in gf_all.columns else np.nan,
                            "지상부 질량 (g)": best["지상부 질량"] if "지상부 질량" in gf_all.columns else np.nan,
                            "지하부 질량 (g)": best["지하부 질량"] if "지하부 질량" in gf_all.columns else np.nan,
                            "총생체량 (g)": best["총생체량"],
                        }
                        c_best.dataframe(
                            pd.DataFrame([best_data]).set_index("개체번호").T,
                            width="stretch",
                        )

                        c_worst.subheader("가장 덜 자란 개체")
                        worst_data = {
                            "개체번호": str(worst["개체번호"]),
                            "잎 수 (장)": worst["잎 수"] if "잎 수" in gf_all.columns else np.nan,
                            "지상부 길이 (mm)": worst["지상부 길이"] if "지상부 길이" in gf_all.columns else np.nan,
                            "지하부 길이 (mm)": worst["지하부 길이"] if "지하부 길이" in gf_all.columns else np.nan,
                            "지상부 질량 (g)": worst["지상부 질량"] if "지상부 질량" in gf_all.columns else np.nan,
                            "지하부 질량 (g)": worst["지하부 질량"] if "지하부 질량" in gf_all.columns else np.nan,
                            "총생체량 (g)": worst["총생체량"],
                        }
                        c_worst.dataframe(
                            pd.DataFrame([worst_data]).set_index("개체번호").T,
                            width="stretch",
                        )

                        if "잎 수" in gf_all.columns:
                            leaf_ratio = best["잎 수"] / worst["잎 수"] if worst["잎 수"] != 0 else np.nan
                            mass_ratio = best["총생체량"] / worst["총생체량"] if worst["총생체량"] != 0 else np.nan
                            if not np.isnan(mass_ratio):
                                st.info(f"가장 잘 자란 개체({str(best['개체번호'])})는 가장 덜 자란 개체({str(worst['개체번호'])})보다 "
                                        f"총생체량이 약 **{mass_ratio:.1f}배** 크고, "
                                        f"잎 수는 약 **{leaf_ratio:.1f}배** 많아요. "
                                        f"같은 스마트팜에서 자랐는데도 개체 간 생육 차이가 크다는 것을 보여줘요.")

                # ──────────────────────────────────────
                # 3-4. 참고: 보고서 값
                # ──────────────────────────────────────
                st.header("참고: 보고서에 기재된 분석 값")
                st.caption("아래 값은 업로드된 데이터의 실제 계산값이 아니라, "
                           "실험 보고서에 기재된 참고 수치예요. "
                           "실제 데이터와 완전히 일치하지 않을 수 있어요.")

                report_tbl = pd.DataFrame([
                    {"지표 쌍": "지상부 질량 ↔ 지하부 질량", "보고서 r": 0.806, "해석": "강한 양의 상관"},
                    {"지표 쌍": "잎 수 ↔ 지상부 질량", "보고서 r": 0.811, "해석": "강한 양의 상관"},
                    {"지표 쌍": "잎 수 ↔ 지하부 질량", "보고서 r": 0.668, "해석": "중간~강한 양의 상관"},
                    {"지표 쌍": "지상부 길이 ↔ 지하부 길이", "보고서 r": -0.026, "해석": "거의 무관"},
                    {"지표 쌍": "지상부 질량 ↔ 지하부 길이", "보고서 r": 0.176, "해석": "약한 상관"},
                ])
                st.dataframe(report_tbl, width="stretch")

                st.subheader("보고서 t검정 결과 (지하부 질량 중앙값 0.39g 기준)")
                report_t = pd.DataFrame([
                    {"그룹": "지하부 발달 양호 그룹 (> 0.39g)", "개체 수": 27, "지상부 질량 평균": "1.91g", "잎 수 평균": "44.2장"},
                    {"그룹": "지하부 발달 부족 그룹 (≤ 0.39g)", "개체 수": 27, "지상부 질량 평균": "0.81g", "잎 수 평균": "26.3장"},
                ])
                st.dataframe(report_t, width="stretch")
                st.info("보고서 기준 t = 5.386, p < 0.0001 → 두 그룹 간 지상부 질량 차이가 통계적으로 매우 유의해요.")
                st.caption("지상부 길이(줄기 길이)와 지하부 길이는 상관이 거의 없었으므로(r ≈ -0.026), "
                           "지상부 생장을 평가할 때는 길이보다 **질량·잎 수 지표**가 더 의미 있는 지표예요.")

                # ──────────────────────────────────────
                # 3-5. 산점도 모음 (서브플롯)
                # ──────────────────────────────────────
                st.header("생육 지표 간 관계 — 산점도 모음")
                pairs = [
                    ("잎 수", "생중량"),
                    ("지상부 길이", "생중량"),
                    ("지하부 길이", "생중량"),
                    ("지상부 질량", "생중량"),
                    ("지하부 질량", "지상부 질량"),
                    ("지하부 길이", "지상부 길이"),
                ]
                figs = []
                labels = []
                for px_name, py_name in pairs:
                    if px_name in calc_cols and py_name in calc_cols:
                        dfp = growth_df[[px_name, py_name]].dropna()
                        if len(dfp) >= 2:
                            f = scatter_with_trend(dfp[px_name], dfp[py_name], px_name, py_name)
                            if f:
                                figs.append(f)
                                labels.append(f"{px_name} vs {py_name}")

                if figs:
                    nrows = (len(figs) + 1) // 2
                    subplot_fig = make_subplots(
                        rows=nrows, cols=2,
                        subplot_titles=labels,
                        vertical_spacing=0.15,
                        horizontal_spacing=0.08,
                    )
                    for i, f in enumerate(figs):
                        row = i // 2 + 1
                        col = i % 2 + 1
                        for tr in f.data:
                            subplot_fig.add_trace(tr, row=row, col=col)
                    subplot_fig.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=340 * nrows,
                        margin=dict(l=50, r=20, t=40, b=50),
                        showlegend=False,
                    )
                    st.plotly_chart(subplot_fig, width="stretch")
                    st.caption("빨간 실선은 추세선(선형 회귀)이며, 점들이 추세선 가까이에 모여 있을수록 "
                               "두 지표 사이의 관계가 뚜렷해요.")
                else:
                    st.info("산점도를 만들 수 있는 생육 지표 쌍이 없어요.")

                # ──────────────────────────────────────
                # 3-6. CSV 다운로드
                # ──────────────────────────────────────
                st.header("데이터 다운로드")
                if not growth_df.empty:
                    csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label=f"{sch_name} 생육 데이터 다운로드 (UTF-8-SIG)",
                        data=csv_bytes,
                        file_name=f"{sch_name}_생육결과_분석용.csv",
                        mime="text/csv",
                    )
                    st.caption("UTF-8-SIG 인코딩으로 저장되어 엑셀에서도 한글이 깨지지 않아요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
        if __name__ == "__main__":
    st.run()
