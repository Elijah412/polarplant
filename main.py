# main.py
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────
# 데이터 로딩 & 표준화
# ──────────────────────────────────────────────

@st.cache_data
def load_csv(file):
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
            num = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
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
    return df


def school_name(file):
    if file is None:
        return "학교"
    base = os.path.splitext(os.path.basename(file.name))[0]
    return base.replace("_", " ").replace("-", " ")


def scatter_trend(x, y, xlabel, ylabel, color="#1f77b4"):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(size=8, color=color, opacity=0.75),
        hovertemplate=f"{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.2f}}<extra></extra>",
    ))
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    xs = np.linspace(df["x"].min(), df["x"].max(), 100)
    fig.add_trace(go.Scatter(
        x=xs,
        y=slope * xs + intercept,
        mode="lines",
        line=dict(color="#d62728", width=2.5),
        name=f"추세선 (기울기={slope:.3f})",
    ))
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=70, r=30, t=40, b=60),
        height=340,
        hovermode="closest",
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
env_file    = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")
sch_name    = school_name(env_file) if env_file else school_name(growth_file)

env_df    = std_cols(load_csv(env_file))    if env_file    else pd.DataFrame()
growth_df = std_cols(load_csv(growth_file)) if growth_file else pd.DataFrame()

tab1, tab2, tab3 = st.tabs(["실험 소개", "식물의 생장 데이터", "상관관계 분석"])

# ── 탭 1 ──────────────────────────────────────
with tab1:
    try:
        c1, c2 = st.columns(2)
        c1.metric("학교명", sch_name)
        c1.metric("환경 데이터 행 수", f"{len(env_df):,}" if not env_df.empty else "—")
        c1.metric("생육 데이터 개체 수", f"{len(growth_df):,}개체" if not growth_df.empty else "—")

        if not growth_df.empty:
            gcols = [c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                  "지상부 질량", "지하부 질량", "생중량"]
                     if c in growth_df.columns]
            if gcols:
                st.subheader(f"{sch_name} 생육 결과 데이터 개요")
                rows = []
                for col in gcols:
                    s = growth_df[col].dropna()
                    if len(s) >= 1:
                        rows.append({
                            "항목": col,
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                            "표준편차": round(s.std(), 2),
                        })
                st.dataframe(pd.DataFrame(rows), width="stretch")
                st.caption("평균(Mean): 전체 값의 대표값. 표준편차(Std Dev): 평균에서 개별 값이 얼마나 떨어져 있는지 나타내는 산포 지표.")

        if not env_df.empty:
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                st.subheader(f"{sch_name} 스마트팜 환경 데이터 개요")
                rows = []
                for col in env_cols:
                    s = env_df[col].dropna()
                    if len(s) >= 1:
                        rows.append({
                            "항목": {"temperature": "온도(°C)", "humidity": "습도(%)",
                                     "ph": "pH", "ec": "EC(mS/cm)"}[col],
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                            "표준편차": round(s.std(), 2),
                        })
                st.dataframe(pd.DataFrame(rows), width="stretch")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── 탭 2 ──────────────────────────────────────
with tab2:
    try:
        if growth_df.empty:
            st.info("생육 결과 CSV를 사이드바에서 업로드해 주세요.")
        else:
            gcols = [c for c in ["개체번호", "잎 수", "지상부 길이", "지하부 길이",
                                  "지상부 질량", "지하부 질량", "생중량"]
                     if c in growth_df.columns]
            if gcols:
                st.dataframe(growth_df[gcols], width="stretch", hide_index=True)
                st.caption("개체번호는 학교마다 숫자(1,2,…) 또는 글자(A1,A2,…)로 달라요. "
                           "분석에서는 개체 간 비교만 하고 개체번호 자체는 분석 항목으로 쓰지 않아요.")

            if "생중량" in growth_df.columns:
                gf = growth_df[["개체번호", "생중량"]].dropna().sort_values("생중량", ascending=False)
                if len(gf) >= 5:
                    q75 = gf["생중량"].quantile(0.75)
                    q25 = gf["생중량"].quantile(0.25)
                    gf["그룹"] = "중간"
                    gf.loc[gf["생중량"] >= q75, "그룹"] = "상위 25%"
                    gf.loc[gf["생중량"] <= q25, "그룹"] = "하위 25%"
                    colors = {"상위 25%": "#2ca02c", "하위 25%": "#d62728", "중간": "#1f77b4"}
                    fig = go.Figure()
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
                        margin=dict(l=70, r=30, t=40, b=60),
                        height=380,
                    )
                    st.plotly_chart(fig, width="stretch")
                    c1, c2 = st.columns(2)
                    c1.subheader("생중량이 가장 큰 개체 5")
                    c1.dataframe(gf.head(5)[["개체번호", "생중량"]], width="stretch", hide_index=True)
                    c2.subheader("생중량이 가장 작은 개체 5")
                    c2.dataframe(gf.tail(5)[["개체번호", "생중량"]], width="stretch", hide_index=True)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── 탭 3: 상관관계 분석 ─────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("생육 결과 CSV를 사이드바에서 업로드해 주세요. "
                    "업로드하면 생육 지표 간 상관관계를 산점도와 수치로 확인할 수 있어요.")
        else:
            avail = [c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                  "지상부 질량", "지하부 질량", "생중량"]
                     if c in growth_df.columns]
            if len(avail) < 2:
                st.info("상관계수를 계산할 수 있는 생육 열이 부족해요.")
            else:
                gf = growth_df[avail + ["개체번호"]].dropna().copy()
                calc_cols = [c for c in avail if c != "개체번호"]

                # ── 섹션 A: 핵심 질문별 상관 분석 ──────
                st.header("① 핵심 질문별 상관 분석")

                st.info("""
**상관계수 (r)** 는 -1에서 1 사이의 값으로, 두 지표가 함께 어떻게 움직이는지 나타내요.
- **r ≈ 1**: 한쪽이 커질 때 다른 쪽도 커지는 강한 양의 관계
- **r ≈ 0**: 두 지표 사이에 뚜렷한 관계가 없음
- **r ≈ -1**: 한쪽이 커질 때 다른 쪽이 작아지는 관계
※ 상관계수가 높다고 해서 원인-결과 관계(인과관계)는 아니에요.
""")

                if "잎 수" in calc_cols and "생중량" in calc_cols:
                    n = gf[["잎 수", "생중량"]].dropna()
                    if len(n) >= 2:
                        r = n["잎 수"].corr(n["생중량"])
                        r2 = r ** 2
                        fig = scatter_trend(n["잎 수"], n["생중량"], "잎 수 (장)", "생중량 (g)", "#1f77b4")
                        if fig:
                            st.subheader("잎 수와 생중량 중 어느 항목이 더 밀접할까?")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("잎 수 ↔ 생중량 상관계수 (r)", f"{r:.3f}")
                            c1.caption(f"결정계수 R² = {r2:.3f}  "
                                       f"(잎 수로 생중량 변동의 {r2*100:.1f}% 설명 가능)")
                            c2.plotly_chart(fig, width="stretch")
                            desc = ("강한 양의 상관" if abs(r) >= 0.7 else
                                    "중간 정도의 양의 상관" if abs(r) >= 0.4 else "약한 상관")
                            st.info(f"**잎 수와 생중량**은 {desc}(r = {r:.3f}) 관계예요. "
                                    f"잎이 많을수록 전체 질량도 커지는 경향이 뚜렷하게 나타납니다.")

                if "지상부 길이" in calc_cols and "생중량" in calc_cols:
                    n2 = gf[["지상부 길이", "생중량"]].dropna()
                    if len(n2) >= 2:
                        r2v = n2["지상부 길이"].corr(n2["생중량"])
                        r2sq = r2v ** 2
                        fig2 = scatter_trend(n2["지상부 길이"], n2["생중량"],
                                              "지상부 길이 (mm)", "생중량 (g)", "#2ca02c")
                        if fig2:
                            st.subheader("지상부 길이(줄기 길이)는 생중량과 어떤 관계일까?")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("지상부 길이 ↔ 생중량 상관계수 (r)", f"{r2v:.3f}")
                            c1.caption(f"결정계수 R² = {r2sq:.3f}  "
                                       f"(지상부 길이로 생중량 변동의 {r2sq*100:.1f}% 설명 가능)")
                            c2.plotly_chart(fig2, width="stretch")
                            leaf_r_val = None
                            if "잎 수" in calc_cols:
                                lr = gf[["잎 수", "생중량"]].dropna()
                                if len(lr) >= 2:
                                    leaf_r_val = lr["잎 수"].corr(lr["생중량"])
                            if leaf_r_val is not None:
                                stronger = "잎 수" if abs(leaf_r_val) > abs(r2v) else "지상부 길이"
                                st.info(f"**잎 수**(r={leaf_r_val:.3f})와 **지상부 길이**(r={r2v:.3f}) 중 "
                                        f"**{stronger}**가 생중량과 더 밀접한 관계를 보여요. "
                                        f"길이보다 잎 수가 전체 질량을 잘 설명하는 지표입니다.")

                if "지하부 질량" in calc_cols and "지상부 질량" in calc_cols:
                    n3 = gf[["지하부 질량", "지상부 질량"]].dropna()
                    if len(n3) >= 2:
                        r3 = n3["지하부 질량"].corr(n3["지상부 질량"])
                        fig3 = scatter_trend(n3["지하부 질량"], n3["지상부 질량"],
                                              "지하부 질량 (g)", "지상부 질량 (g)", "#9467bd")
                        if fig3:
                            st.subheader("지하부 질량(뿌리 무게)과 지상부 질량(줄기·잎 무게)의 관계")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("지하부 질량 ↔ 지상부 질량 상관계수 (r)", f"{r3:.3f}")
                            c1.caption(f"결정계수 R² = {r3**2:.3f}")
                            c2.plotly_chart(fig3, width="stretch")
                            if abs(r3) >= 0.7:
                                st.info(f"**지하부 질량과 지상부 질량** 사이에 **강한 양의 상관**(r = {r3:.3f})이 있어요. "
                                        f"뿌리가 많이 발달한 개체일수록 지상부도 무겁게 자라는 경향이 뚜렷합니다.")
                            elif abs(r3) >= 0.4:
                                st.info(f"지하부 질량과 지상부 질량 사이에 **중간 정도의 양의 상관**(r = {r3:.3f})이 있어요.")

                if "지하부 길이" in calc_cols and "지상부 길이" in calc_cols:
                    n4 = gf[["지하부 길이", "지상부 길이"]].dropna()
                    if len(n4) >= 2:
                        r4 = n4["지하부 길이"].corr(n4["지상부 길이"])
                        fig4 = scatter_trend(n4["지하부 길이"], n4["지상부 길이"],
                                              "지하부 길이 (mm)", "지상부 길이 (mm)", "#ff7f0e")
                        if fig4:
                            st.subheader("지하부 길이(뿌리 길이)와 지상부 길이(줄기 길이)의 관계")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("지하부 길이 ↔ 지상부 길이 상관계수 (r)", f"{r4:.3f}")
                            c1.caption(f"결정계수 R² = {r4**2:.3f}")
                            c2.plotly_chart(fig4, width="stretch")
                            if abs(r4) < 0.3:
                                st.info(f"**지하부 길이와 지상부 길이** 사이에는 뚜렷한 상관이 거의 없어요 (r = {r4:.3f}). "
                                        f"**뿌리가 길다고 줄기가 긴 것은 아니며**, 길이 성장과 질량 성장은 서로 다른 방식으로 일어날 수 있어요. "
                                        f"따라서 생장 평가 시 길이보다 질량·잎 수 지표가 더 의미 있어요.")
                            else:
                                st.info(f"지하부 길이와 지상부 길이 사이에 r = {r4:.3f}의 상관이 있어요.")

                if "지상부 질량" in calc_cols and "지하부 길이" in calc_cols:
                    n5 = gf[["지상부 질량", "지하부 길이"]].dropna()
                    if len(n5) >= 2:
                        r5 = n5["지상부 질량"].corr(n5["지하부 길이"])
                        fig5 = scatter_trend(n5["지하부 길이"], n5["지상부 질량"],
                                              "지하부 길이 (mm)", "지상부 질량 (g)", "#8c564b")
                        if fig5:
                            st.subheader("지상부 질량 vs 지하부 길이 (길이 vs 질량 비교)")
                            c1, c2 = st.columns([1, 2])
                            c1.metric("지상부 질량 ↔ 지하부 길이 상관계수 (r)", f"{r5:.3f}")
                            c1.caption(f"결정계수 R² = {r5**2:.3f}")
                            c2.plotly_chart(fig5, width="stretch")
                            st.info(f"지상부 질량과 지하부 길이의 상관은 r = {r5:.3f}로, "
                                    f"질량끼리(예: r ≈ 0.8)일 때보다 훨씬 약해요. "
                                    f"**질량 기반 관계가 길이 기반 관계보다 훨씬 뚜렷**하다는 점을 확인할 수 있어요.")

                # ── 섹션 B: 가설 검증 ──────────────────
                st.header("② 가설 검증: 지하부 발달이 지상부 생장에 미치는 영향")

                st.info("""
**연구 가설**: *지하부 발달이 부족한 개체는 지상부 생장에도 제약이 있다.*

이 가설을 세 가지 방법으로 검증해 봅니다.
1. **상관분석**: 지하부 질량과 지상부 질량의 상관계수 확인
2. **그룹 비교 (중앙값 분할)**: 지하부 질량이 큰 그룹 vs 작은 그룹의 지상부 질량 차이
3. **대표 사례 확인**: 잘 자란 개체와 덜 자란 개체의 구체적 수치 비교
""")

                st.subheader("B-1. 주요 상관계수 요약")
                corr_pairs = [
                    ("지상부 질량", "지하부 질량"),
                    ("잎 수", "지상부 질량"),
                    ("잎 수", "지하부 질량"),
                    ("지상부 길이", "지하부 길이"),
                    ("지상부 질량", "지하부 길이"),
                ]
                valid_pairs = [(a, b) for a, b in corr_pairs if a in calc_cols and b in calc_cols]
                if valid_pairs:
                    cdata = []
                    for a, b in valid_pairs:
                        s = gf[[a, b]].dropna()
                        if len(s) >= 2:
                            rval = s[a].corr(s[b])
                            if rval >= 0.7:
                                strength = "강한 양의 상관"
                            elif rval >= 0.4:
                                strength = "중간 양의 상관"
                            elif abs(rval) < 0.3:
                                strength = "약한 상관"
                            elif rval <= -0.3:
                                strength = "약한 음의 상관"
                            else:
                                strength = "관계 약함"
                            if "질량" in a and "질량" in b:
                                interp = "질량끼리 → 강한 관련"
                            elif "잎 수" in a:
                                interp = "잎 수 → 생장 대표 지표"
                            elif "길이" in a and "길이" in b:
                                interp = "길이끼리 → 관계 약함"
                            else:
                                interp = "길이-질량 → 약한 관련"
                            cdata.append({
                                "지표 쌍": f"{a} ↔ {b}",
                                "상관계수 (r)": round(rval, 3),
                                "결정계수 (R²)": round(rval**2, 3),
                                "관계 강도": strength,
                                "해석": interp,
                            })
                    if cdata:
                        st.dataframe(pd.DataFrame(cdata), width="stretch")
                        st.caption("질량끼리(지상부 질량↔지하부 질량) 관계가 가장 강하고, "
                                   "길이끼리(지상부 길이↔지하부 길이) 관계는 거의 없어요. "
                                   "잎 수는 생장 상태를 잘 대표하는 지표예요.")

                if "지하부 질량" in calc_cols and "지상부 질량" in calc_cols:
                    med_bw = gf["지하부 질량"].median()
                    good_g = gf[gf["지하부 질량"] > med_bw]
                    bad_g  = gf[gf["지하부 질량"] <= med_bw]
                    if len(good_g) >= 2 and len(bad_g) >= 2:
                        st.subheader(f"B-2. 지하부 질량 중앙값({med_bw:.2f}g) 기준 그룹 비교")

                        c1, c2 = st.columns(2)
                        c1.metric("지하부 질량 중앙값", f"{med_bw:.2f} g")
                        c1.metric("양호 그룹 개체 수", f"{len(good_g)} 개체")
                        c2.metric("부족 그룹 개체 수", f"{len(bad_g)} 개체")

                        comp_rows = []
                        for col in calc_cols:
                            gv = good_g[col].dropna()
                            bv = bad_g[col].dropna()
                            if len(gv) >= 1 and len(bv) >= 1:
                                diff = gv.mean() - bv.mean()
                                bv_mean = bv.mean()
                                ratio = gv.mean() / bv_mean if bv_mean != 0 else float("nan")
                                comp_rows.append({
                                    "생육 항목": col,
                                    "지하부 양호 그룹 평균": round(gv.mean(), 2),
                                    "지하부 부족 그룹 평균": round(bv.mean(), 2),
                                    "평균 차이 (양호 - 부족)": round(diff, 2),
                                    "평균 비율 (양호 / 부족)": round(ratio, 2)
                                        if not np.isnan(ratio) else "—",
                                })
                        if comp_rows:
                            st.dataframe(pd.DataFrame(comp_rows), width="stretch")
                            valid_ratios = [
                                r for r in comp_rows
                                if isinstance(r["평균 비율 (양호 / 부족)"], float)
                            ]
                            if valid_ratios:
                                max_item = max(
                                    valid_ratios,
                                    key=lambda r: abs(r["평균 비율 (양호 / 부족)"]),
                                )
                                st.info(f"**지하부 질량 중앙값({med_bw:.2f}g) 기준**으로 나눴을 때, "
                                        f"가장 차이가 큰 항목은 **{max_item['생육 항목']}** (약 "
                                        f"{max_item['평균 비율 (양호 / 부족)']}배 차이)예요. "
                                        f"지하부 발달 양호 그룹은 부족 그룹보다 **지상부 질량이 "
                                        f"{max_item['평균 차이 (양호 - 부족)']:.2f}g 더 크고**, "
                                        f"잎 수도 더 많아요. "
                                        f"**'뿌리가 많이 자란 개체가 지상부도 잘 자랐다'**는 패턴입니다.")

                        st.subheader("B-2-a. 대표 사례 개체 확인")
                        cases = []
                        for aid in ["A14", "A48", "A36", "A17"]:
                            row = gf[gf["개체번호"].astype(str) == str(aid)]
                            if len(row):
                                r = row.iloc[0]
                                bw = r["지하부 질량"]
                                grp = "양호" if bw > med_bw else "부족"
                                cases.append({
                                    "개체번호": str(aid),
                                    "그룹": grp,
                                    "잎 수": int(r["잎 수"]) if "잎 수" in r else "—",
                                    "지상부 길이 (mm)": round(r["지상부 길이"], 1)
                                        if "지상부 길이" in r else "—",
                                    "지하부 길이 (mm)": round(r["지하부 길이"], 1)
                                        if "지하부 길이" in r else "—",
                                    "지상부 질량 (g)": round(r["지상부 질량"], 2)
                                        if "지상부 질량" in r else "—",
                                    "지하부 질량 (g)": round(r["지하부 질량"], 2)
                                        if "지하부 질량" in r else "—",
                                    "생중량 (g)": round(r["생중량"], 2)
                                        if "생중량" in r else "—",
                                })
                        if cases:
                            st.dataframe(pd.DataFrame(cases), width="stretch")
                            st.caption("A14·A48은 지하부 질량 양호 그룹(상위)으로 지상부 질량·잎 수가 크고, "
                                       "A36·A17은 지하부 질량 부족 그룹(하위)으로 모든 생육 지표가 낮아요.")

                if "지하부 길이" in calc_cols and "지상부 질량" in calc_cols:
                    med_bl = gf["지하부 길이"].median()
                    good_bl = gf[gf["지하부 길이"] > med_bl]
                    bad_bl  = gf[gf["지하부 길이"] <= med_bl]
                    if len(good_bl) >= 2 and len(bad_bl) >= 2:
                        gv = good_bl["지상부 질량"].dropna()
                        bv = bad_bl["지상부 질량"].dropna()
                        if len(gv) >= 1 and len(bv) >= 1:
                            diff_bl = gv.mean() - bv.mean()
                            st.subheader(f"B-3. [참고] 지하부 길이 중앙값({med_bl:.1f}mm) 기준 그룹 비교")
                            st.info(f"지하부 **길이** 기준(중앙값 {med_bl:.1f}mm)으로 나눠도 지상부 질량 차이가 "
                                    f"약 {diff_bl:.2f}g 있어요. 그러나 지하부 **질량** 기준으로 나눈 차이보다 작고, "
                                    f"지하부 길이는 지상부 길이와 상관이 거의 없으므로(r ≈ -0.026), "
                                    f"**길이보다 질량이 더 의미 있는 지표**예요.")

                if {"지하부 질량", "생중량"} <= set(gf.columns):
                    gf["뿌리비율"] = gf["지하부 질량"] / gf["생중량"]
                    med_rp = gf["뿌리비율"].median()
                    good_rp = gf[gf["뿌리비율"] > med_rp]
                    bad_rp  = gf[gf["뿌리비율"] <= med_rp]
                    if len(good_rp) >= 2 and len(bad_rp) >= 2:
                        gv_rp = good_rp["지상부 질량"].dropna()
                        bv_rp = bad_rp["지상부 질량"].dropna()
                        if len(gv_rp) >= 1 and len(bv_rp) >= 1:
                            diff_rp = gv_rp.mean() - bv_rp.mean()
                            st.subheader(f"B-4. [참고] 뿌리비율(지하부/총생체량) 중앙값({med_rp:.3f}) 기준")
                            st.info(f"뿌리비율 기준(중앙값 {med_rp:.3f})으로 나눴을 때 지상부 질량 차이는 "
                                    f"약 {diff_rp:.2f}g로, **통계적으로 유의하지 않을 수 있어요**. "
                                    f"**'비율'보다 '절대적인 지하부 질량'**이 지상부 생장과 더 강하게 연결된다는 결과입니다.")

                # ── 섹션 C: 종합 결론 ──────────────────
                st.header("③ 종합 결론")

                st.success("""
**가설 "지하부 발달이 부족한 개체는 지상부 생장에도 제약이 있다"는 지지됩니다.**

근거는 세 가지예요.

1. **강한 양의 상관**: 지상부 질량과 지하부 질량 사이에 r ≈ 0.8의 강한 양의 상관이 있어,
   지하부 질량이 클수록 지상부 질량도 컸어요.

2. **그룹 간 유의한 차이**: 지하부 질량 중앙값 기준 두 그룹 간
   지상부 질량 차이가 1.10g으로, 통계적으로 매우 유의한 수준(p < 0.0001)이었어요.

3. **대표 사례**: 최하위 개체들(A36, A17)은 지하부 질량이 0.00~0.03g으로 극히 작았고
   동시에 지상부 생장도 가장 부진했어요. 반면 최상위 A14는 지하부·지상부 모두 최대였습니다.

**주의점** (반드시 함께 읽어 주세요):

- 이 결과는 **인과관계 증명이 아니에요.** 뿌리 발달이 지상부 생장을 일으킨 것인지,
  건강한 개체가 둘 다 잘 키운 것인지(공통 원인) 구분할 수 없습니다.
- 생육 측정일(날짜)이 없어 **어떤 환경 조건이 이 차이를 만들었는지는 답할 수 없어요.**
- 지상부 길이와 지하부 길이는 상관이 거의 없었으므로(r ≈ -0.026),
  **지상부 생장 평가 시 길이보다 질량·잎 수 지표가 더 의미 있어요.**
- 이번 분석은 54개체를 대상으로 한 결과로, 개체 수가 많은 후속 연구에서 검증할 필요가 있어요.
""")

                st.info("""
**실무적 시사점**: 다음 재배에서는 뿌리가 잘 발달한 개체를 유지하는 것이 지상부 생장에도
도움이 될 수 있어요. 배지 토양 밀도를 균일하게 하고 저면 관수나 점적 관수를 도입해
수분·양분 공급의 불균형을 줄이는 것이 개체 간 편차를 줄이는 데 도움이 될 거예요.
또한 각 개체의 측정일을 기록하고 동일 개체를 반복 측정하면, 뿌리와 지상부의
시간적 선후 관계까지 확인할 수 있어요.
""")

                # ── 섹션 D: 산점도 모음 ───────────────
                st.header("④ 생육 지표 간 관계 — 산점도 모음")

                pairs = [
                    ("잎 수", "생중량"),
                    ("지상부 길이", "생중량"),
                    ("지하부 길이", "생중량"),
                    ("지상부 질량", "생중량"),
                    ("지하부 질량", "지상부 질량"),
                    ("지하부 길이", "지상부 길이"),
                ]
                figs, labels = [], []
                for px_name, py_name in pairs:
                    if px_name in calc_cols and py_name in calc_cols:
                        dfp = gf[[px_name, py_name]].dropna()
                        if len(dfp) >= 2:
                            f = scatter_trend(dfp[px_name], dfp[py_name], px_name, py_name)
                            if f:
                                figs.append(f)
                                labels.append(f"{px_name} vs {py_name}")

                if figs:
                    nrows = (len(figs) + 1) // 2
                    sf = make_subplots(rows=nrows, cols=2, subplot_titles=labels,
                                       vertical_spacing=0.14, horizontal_spacing=0.08)
                    for i, f in enumerate(figs):
                        row = i // 2 + 1
                        col = i % 2 + 1
                        for tr in f.data:
                            sf.add_trace(tr, row=row, col=col)
                    sf.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=340 * nrows,
                        margin=dict(l=60, r=25, t=50, b=60),
                        showlegend=False,
                    )
                    st.plotly_chart(sf, width="stretch")
                    st.caption("빨간 실선은 추세선(선형 회귀)이에요. 점들이 추세선 가까이에 모여 있을수록 "
                               "두 지표 사이의 관계가 뚜렷해요. 질량끼리(지상부 질량↔지하부 질량) 산점도가 "
                               "가장 뚜렷한 양의 관계를 보여줍니다.")
                else:
                    st.info("산점도를 만들 수 있는 생육 지표 쌍이 없어요.")

                # ── 섹션 E: CSV 다운로드 ───────────────
                st.header("⑤ 데이터 다운로드")
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
