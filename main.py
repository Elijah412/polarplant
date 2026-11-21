# main.py
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    name = base.replace("_", " ").replace("-", " ")
    return name

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

env_raw   = load_csv(env_file)
growth_raw = load_csv(growth_file)
env_df    = std_cols(env_raw) if not env_raw.empty else pd.DataFrame()
growth_df = std_cols(growth_raw) if not growth_raw.empty else pd.DataFrame()

sch_name  = school_name(env_file) if not env_raw.empty else school_name(growth_file)

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
                st.caption(f"개체번호는 학교마다 숫자(1, 2, …) 또는 글자(A1, A2, …)로 달라요. "
                           f"분석에서는 개체 간 비교만 하고 개체번호 자체는 분석 항목으로 쓰지 않아요.")

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
            # ── 상관계수 설명 카드
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

                # ── 1) 질문별 상관 분석
                st.header("핵심 질문과 상관 분석 결과")

                # 질문 1: 잎 수 vs 생중량 vs 지상부 길이 vs 생중량
                if "잎 수" in calc_cols and "생중량" in calc_cols:
                    n1 = growth_df[["잎 수", "생중량"]].dropna()
                    if len(n1) >= 2:
                        r1 = n1["잎 수"].corr(n1["생중량"])
                        r1sq = r1 ** 2
                        fig1 = scatter_with_trend(n1["잎 수"], n1["생중량"], "잎 수(장)", "생중량(g)")
                        if fig1:
                            st.subheader("① 잎 수와 생중량 중 어느 것이 더 밀접할까?")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("잎 수 ↔ 생중량 상관계수 (r)", f"{r1:.3f}")
                            c1.caption(f"결정계수 R² = {r1sq:.3f} "
                                       f"(잎 수로 생중량 변동의 {r1sq*100:.1f}% 설명 가능)")
                            c2.plotly_chart(fig1, width="stretch")
                            desc = ("강한 양의 상관" if abs(r1) >= 0.7 else
                                    "중간 정도의 양의 상관" if abs(r1) >= 0.4 else "약한 상관")
                            st.info(f"잎 수와 생중량은 {desc} (r = {r:.3f}) 관계를 보입니다. "
                                    f"잎이 많을수록 전체 질량도 커지는 경향이 있어요.")

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
                            c1.caption(f"결정계수 R² = {r2sq:.3f} "
                                       f"(지상부 길이로 생중량 변동의 {r2sq*100:.1f}% 설명 가능)")
                            c2.plotly_chart(fig2, width="stretch")
                            desc2 = ("강한 양의 상관" if abs(r2) >= 0.7 else
                                     "중간 정도의 양의 상관" if abs(r2) >= 0.4 else "약한 상관")
                            cmp = ("잎 수가" if ("잎 수" in calc_cols and len(n1) >= 2 and abs(r1) > abs(r2))
                                   else "지상부 길이가") + f" 생중량과 더 밀접한 관계(r={max(abs(r1), abs(r2)):.3f})를 보입니다."
                            st.info(cmp)

                # 질문 2: 상·하위 그룹 비교
                if "생중량" in calc_cols:
                    gf = growth_df[["개체번호", "생중량"] + [c for c in calc_cols if c not in ("개체번호", "생중량")]].dropna()
                    if len(gf) >= 5:
                        q75 = gf["생중량"].quantile(0.75)
                        q25 = gf["생중량"].quantile(0.25)
                        top = gf[gf["생중량"] >= q75]
                        bot = gf[gf["생중량"] <= q25]
                        if len(top) >= 2 and len(bot) >= 2:
                            st.subheader("② 무거운 개체와 가벼운 개체는 어떤 생육 특징에서 차이가 날까?")
                            comp_rows = []
                            for col in [c for c in calc_cols if c not in ("개체번호", "생중량")]:
                                tv = top[col].dropna()
                                bv = bot[col].dropna()
                                if len(tv) >= 1 and len(bv) >= 1:
                                    diff = tv.mean() - bv.mean()
                                    ratio = tv.mean() / bv.mean() if bv.mean() != 0 else float("nan")
                                    comp_rows.append({
                                        "생육 항목": col,
                                        "상위 25% 평균": round(tv.mean(), 2),
                                        "하위 25% 평균": round(bv.mean(), 2),
                                        "평균 차이 (상위-하위)": round(diff, 2),
                                        "평균 비율 (상위/하위)": round(ratio, 2) if not np.isnan(ratio) else "—",
                                    })
                            if comp_rows:
                                comp_df = pd.DataFrame(comp_rows)
                                st.dataframe(comp_df, width="stretch")
                                max_item = max(comp_rows, key=lambda r: abs(r["평균 비율 (상위/하위)"]) if isinstance(r["평균 비율 (상위/하위)"], float) else 0)
                                st.info(f"상위 25%와 하위 25% 그룹 간 차이가 가장 큰 항목은 "
                                        f"**{max_item['생육 항목']}** (약 {max_item['평균 비율 (상위/하위)']}배 차이)예요. "
                                        f"이 항목이 잘 자란 개체와 덜 자란 개체를 가르는 핵심 지표일 가능성이 높아요.")

                # 질문 3: 지하부 생장이 지상부 생장에 미치는 영향
                if "지하부 질량" in calc_cols and "지상부 질량" in calc_cols:
                    n3 = growth_df[["지하부 질량", "지상부 질량"]].dropna()
                    if len(n3) >= 2:
                        r3 = n3["지하부 질량"].corr(n3["지상부 질량"])
                        fig3 = scatter_with_trend(n3["지하부 질량"], n3["지상부 질량"],
                                                   "지하부 질량(g)", "지상부 질량(g)")
                        if fig3:
                            st.subheader("③ 지하부 생장이 지상부 생장에 어떤 영향을 미칠까?")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("지하부 질량 ↔ 지상부 질량 상관계수 (r)", f"{r3:.3f}")
                            c1.caption(f"결정계수 R² = {r3**2:.3f}")
                            c2.plotly_chart(fig3, width="stretch")
                            if abs(r3) >= 0.7:
                                st.info("지하부 질량과 지상부 질량 사이에 **강한 양의 상관**(r ≈ %.3f)이 있어요. "
                                        "뿌리가 많이 발달한 개체일수록 줄기와 잎도 무겁게 자라는 경향이 뚜렷해요." % r3)
                            elif abs(r3) >= 0.4:
                                st.info("지하부 질량과 지상부 질량 사이에 **중간 정도의 양의 상관**(r ≈ %.3f)이 있어요. "
                                        "뿌리 발달이 지상부 생장과 어느 정도 함께 움직이지만, 다른 요인도 작용할 수 있어요." % r3)
                            else:
                                st.info("지하부 질량과 지상부 질량 사이에 뚜렷한 상관이 보이지 않아요 (r ≈ %.3f). "
                                        "길이와 질량은 서로 다른 방식으로 자랄 수 있어요." % r3)

                if "지하부 길이" in calc_cols and "지상부 길이" in calc_cols:
                    n4 = growth_df[["지하부 길이", "지상부 길이"]].dropna()
                    if len(n4) >= 2:
                        r4 = n4["지하부 길이"].corr(n4["지상부 길이"])
                        fig4 = scatter_with_trend(n4["지하부 길이"], n4["지상부 길이"],
                                                   "지하부 길이(mm)", "지상부 길이(mm)")
                        if fig4:
                            c1, c2 = st.columns([1, 2])
                            c1.metric("지하부 길이 ↔ 지상부 길이 상관계수 (r)", f"{r4:.3f}")
                            c1.caption(f"결정계수 R² = {r4**2:.3f}")
                            c2.plotly_chart(fig4, width="stretch")
                            if abs(r4) < 0.3:
                                st.info("지하부 길이와 지상부 길이 사이에는 뚜렷한 상관이 거의 없어요 (r ≈ %.3f). "
                                        "뿌리가 길다고 줄기(지상부)가 긴 것은 아니며, 길이 성장과 질량 성장은 서로 다른 양상일 수 있어요." % r4)
                            else:
                                st.info("지하부 길이와 지상부 길이 사이에 r ≈ %.3f의 상관이 있어요." % r4)

                # ── 2) 산점도 모음 (서브플롯)
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

                # ── 3) CSV 다운로드
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
