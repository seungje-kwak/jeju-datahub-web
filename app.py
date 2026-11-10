# -*- coding: utf-8 -*-
"""
app.py - 제주데이터허브 전용 AI 멀티모달(CLIP) 이미지 검색 시스템
(Pinterest / Google Images 스타일 웹 대시보드)

경영정보시스템연구 중간고사 대체과제:
- Multilingual CLIP (sentence-transformers/clip-ViT-B-32-multilingual-v1) 한국어 텍스트 임베딩
- Vision CLIP (openai/clip-vit-base-patch32) 사전 계산 벡터 캐시
- KSIC 표준산업분류 동의어 질의 확장 & Explainable AI 일치 이유 설명
- 실시간 유사도 임계치(Threshold) 필터 슬라이더
- Streamlit Community Cloud 최적화 (초경량 부팅, 캐싱, 상대 경로 지원)
"""

import io
import os
import sys
import json
import html
import streamlit as st
from PIL import Image
import qrcode

# 패키지 경로 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from searcher import (
    JejuMultimodalSearcher,
    DATA_DIR,
    PAGES_INDEX_PATH,
    EMBEDDINGS_PATH,
    META_PATH
)
from collector import fetch_and_download_reports
from extractor import extract_pages_and_render_images

st.set_page_config(
    page_title="제주데이터허브 차트 & 이미지 검색",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pinterest & Google Images 스타일 모던 CSS
st.markdown("""
<style>
    /* 전체 여백 및 폰트 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }
    .header-box {
        text-align: center;
        margin-bottom: 1.6rem;
    }
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1A365D, #2B6CB0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
        letter-spacing: -0.5px;
    }
    .header-desc {
        color: #718096;
        font-size: 1.05rem;
    }
    /* 카드 디자인 */
    .image-card {
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 18px;
    }
    .image-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #3182CE;
    }
    .card-meta {
        padding: 10px 12px 12px 12px;
    }
    .card-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #2D3748;
        line-height: 1.35;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 2.4em;
    }
    .card-sub {
        font-size: 0.76rem;
        color: #A0AEC0;
        margin-top: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .score-chip {
        background-color: #EBF8FF;
        color: #2B6CB0;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.75rem;
        border: 1px solid #BEE3F8;
    }
    .score-chip-high {
        background-color: #DEF7EC;
        color: #03543F;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.75rem;
        border: 1px solid #BCF0DA;
    }
    .tag-container {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    /* 하이라이트 배지 */
    .highlight-match {
        background-color: #FEFCBF;
        font-weight: bold;
        color: #744210;
        padding: 1px 4px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_searcher():
    """검색 엔진 싱글톤 로드 (Streamlit 리소스 캐싱)"""
    return JejuMultimodalSearcher()


def resolve_image_path(item: dict) -> str:
    """Windows 로컬 및 Linux(Streamlit Cloud) 환경 모두에서 이미지 경로 동적 해결"""
    img_name = item.get("image_filename")
    if not img_name and item.get("image_path"):
        img_name = os.path.basename(item["image_path"])

    if img_name:
        candidate = os.path.join(DATA_DIR, "images", img_name)
        if os.path.exists(candidate):
            return candidate
        alt_candidate = os.path.join(CURRENT_DIR, "data", "images", img_name)
        if os.path.exists(alt_candidate):
            return alt_candidate

    raw_path = item.get("image_path")
    if raw_path and os.path.exists(raw_path):
        return raw_path

    return None


def resolve_pdf_path(item: dict) -> str:
    """로컬 PDF 파일 경로 동적 해결"""
    pdf_name = item.get("pdf_filename")
    if not pdf_name and item.get("pdf_path"):
        pdf_name = os.path.basename(item["pdf_path"])

    if pdf_name:
        candidate = os.path.join(DATA_DIR, "reports", pdf_name)
        if os.path.exists(candidate):
            return candidate
        alt_candidate = os.path.join(CURRENT_DIR, "data", "reports", pdf_name)
        if os.path.exists(alt_candidate):
            return alt_candidate

    raw_path = item.get("pdf_path")
    if raw_path and os.path.exists(raw_path):
        return raw_path

    return None


def get_chart_title(item: dict) -> str:
    """페이지 텍스트에서 실제 차트 세부 제목([국내관광객] 비교 차트 등) 추출"""
    text = item.get("full_text") or item.get("text") or item.get("snippet") or ""
    for line in text.split("\n"):
        line = line.strip()
        if "[" in line and "]" in line and len(line) > 5 and not line.startswith("※"):
            return line
    return item.get("report_title", "제주 소비동향 통계 차트")


def highlight_text(snippet: str, query: str) -> str:
    """검색 키워드 및 동의어 노란색 하이라이팅 HTML 태그 적용"""
    if not snippet:
        return ""
    escaped = html.escape(snippet)
    try:
        s = get_searcher()
        terms = s.expand_query_terms(query)
    except Exception:
        terms = [w.strip() for w in query.split() if len(w.strip()) > 0]

    for w in sorted(set(terms), key=len, reverse=True):
        if len(w) >= 2:
            esc_w = html.escape(w)
            escaped = escaped.replace(
                esc_w,
                f'<span class="highlight-match">{esc_w}</span>'
            )
    return escaped


def get_match_reason(item: dict, query: str) -> str:
    """차트와 검색어가 일치하게 된 구체적인 이유 설명 생성 (Explainable AI)"""
    matched = item.get("matched_term", "")
    score_pct = int(item.get("final_score", 0.0) * 100)
    q = query.strip()
    if matched:
        if matched.lower() in q.lower() or q.lower() in matched.lower():
            return f"차트 항목 및 분석 내용에 '{matched}' 키워드가 직접 수록되어 {score_pct}% 일치"
        else:
            return f"검색어 '{q}'의 통계 업종명인 '{matched}' 차트 항목과 매칭되어 {score_pct}% 일치"
    elif item.get("keyword_score", 0) > 0:
        return f"차트 본문 텍스트 내 키워드 문맥 일치 ({score_pct}%)"
    else:
        return f"차트 시각 구조 및 소비 빅데이터 패턴 유사도 판정 ({score_pct}%)"


# 고해상도 이미지 라이트박스 팝업 (Streamlit Dialog)
@st.dialog("📊 차트 상세 보기", width="large")
def show_image_modal(item: dict):
    img_path = resolve_image_path(item)
    if img_path:
        st.image(img_path, use_container_width=True)

    chart_title = get_chart_title(item)
    st.markdown(f"### {chart_title}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"📁 **원본 리포트:** {item.get('report_title')}")
        st.write(f"📅 **발간일:** {item.get('report_date', 'N/A')}")
        st.write(f"📄 **페이지:** {item.get('page_number')} / {item.get('total_pages', '?')} 쪽")
    with col2:
        if "final_score" in item:
            st.metric("종합 일치율", f"{item['final_score']*100:.1f}%")
        if item.get("matched_term"):
            st.write(f"🏷️ **매칭된 업종명:** `{item['matched_term']}`")

    if item.get("snippet"):
        st.info(f"**해당 페이지 핵심 요약:**\n{item['snippet']}")

    st.markdown("---")
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as file:
                st.download_button(
                    label="💾 차트 이미지 다운로드 (PNG)",
                    data=file,
                    file_name=item.get("image_filename", "jeju_chart.png"),
                    mime="image/png",
                    use_container_width=True
                )
    with btn_c2:
        local_pdf = resolve_pdf_path(item)
        if local_pdf and os.path.exists(local_pdf):
            with open(local_pdf, "rb") as pdf_file:
                st.download_button(
                    label="📥 원본 리포트 PDF 다운로드",
                    data=pdf_file,
                    file_name=item.get("pdf_filename", "report.pdf"),
                    mime="application/pdf",
                    use_container_width=True
                )
        elif item.get("file_info_id") and item.get("file_id"):
            dl_url = f"https://jejudatahub.net/api/file?id={item['file_info_id']}&fileId={item['file_id']}"
            st.link_button("🌐 제주데이터허브에서 원본 PDF 받기", dl_url, use_container_width=True)


def generate_qr_image(url: str) -> bytes:
    """URL에 대한 QR 코드 이미지 바이트 생성"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1A365D", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    # 상단 헤더
    st.markdown(
        '<div class="header-box">'
        '<div class="header-title">🌴 제주데이터허브 차트 & 이미지 검색</div>'
        '<div class="header-desc">원하는 키워드를 입력하면 빅데이터 소비 트렌드 차트 및 그래프 이미지만 즉시 찾아드립니다.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 갤러리 설정")

        report_count = 0
        page_count = 0
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                report_count = len(json.load(f))
        if os.path.exists(PAGES_INDEX_PATH):
            with open(PAGES_INDEX_PATH, "r", encoding="utf-8") as f:
                page_count = len(json.load(f))

        st.metric(label="색인된 리포트", value=f"{report_count} 편")
        st.metric(label="보유 차트/페이지", value=f"{page_count} 장")

        st.markdown("---")
        st.subheader("📱 스마트폰 접속 QR")
        default_app_url = "https://jeju-datahub-image-search.streamlit.app"
        app_url_input = st.text_input(
            "배포 URL (스마트폰 접속용)",
            value=default_app_url,
            help="Streamlit Cloud에 배포한 공개 URL을 입력하면 실시간으로 QR 코드가 갱신됩니다."
        )

        qr_bytes = generate_qr_image(app_url_input)
        st.image(qr_bytes, caption="카메라로 스캔하면 스마트폰에서 즉시 열립니다", width=220)
        st.download_button(
            label="💾 QR 코드 다운로드 (PNG)",
            data=qr_bytes,
            file_name="jeju_search_qr.png",
            mime="image/png",
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("🎛️ 레이아웃 및 검색 파라미터")
        grid_cols = st.radio("화면 배열 열 수", [3, 4], index=1, horizontal=True)
        top_k = st.slider("출력할 최대 이미지 수", min_value=6, max_value=36, value=16, step=2)
        alpha = st.slider(
            "시각/의미 유사도 가중치 (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
            help="1.0에 가까울수록 CLIP 멀티모달 시각 점수를 우선하며, 0에 가까울수록 키워드 매칭을 우선합니다."
        )

        st.markdown("---")
        with st.expander("ℹ️ 과제 정보 및 시스템 사양"):
            st.markdown("""
            - **과목:** 경영정보시스템연구 (김근형 교수님)
            - **과제:** 중간고사 대체과제
            - **도구:** Antigravity 2.0 (Google DeepMind)
            - **임베딩 모델:**
              - Text: `clip-ViT-B-32-multilingual-v1`
              - Vision: `openai/clip-vit-base-patch32`
            - **특징:** KSIC 동의어 확장 + Explainable AI 일치 이유 제공
            """)

        st.markdown("---")
        if st.button("🔄 최신 데이터 다시 수집/인덱싱", use_container_width=True):
            with st.spinner("제주데이터허브에서 최신 리포트를 수집하고 인덱싱하는 중..."):
                fetch_and_download_reports(max_reports=10)
                extract_pages_and_render_images(dpi=150)
                searcher = get_searcher()
                searcher.build_embeddings()
                st.cache_resource.clear()
            st.success("데이터 갱신 완료!")
            st.rerun()

    if not os.path.exists(EMBEDDINGS_PATH):
        st.warning("⚠️ 색인 데이터가 없습니다. 사이드바에서 [최신 데이터 다시 수집/인덱싱]을 실행해 주세요.")
        return

    searcher = get_searcher()

    # 인기 추천 검색어 태그 버튼 (동의어 사전 연동)
    sample_tags = ["전체보기", "렌터카", "외식 (음식점)", "골프장", "숙박 (호텔)", "카페·음료", "면세점 쇼핑", "주유소"]
    cols_tag = st.columns(len(sample_tags))
    clicked_tag = None
    for idx, tag in enumerate(sample_tags):
        clean_tag = tag.split()[0].replace("·음료", "")
        if cols_tag[idx].button(tag, key=f"tag_{idx}", use_container_width=True):
            clicked_tag = clean_tag

    # 세션 상태 기반 쿼리 관리
    if "current_query" not in st.session_state:
        st.session_state["current_query"] = ""

    if clicked_tag:
        st.session_state["current_query"] = "" if clicked_tag == "전체보기" else clicked_tag

    # 검색 입력창
    query = st.text_input(
        "검색어 입력",
        value=st.session_state["current_query"],
        placeholder="검색할 키워드를 입력하고 엔터(Enter)를 누르세요 (예: 렌터카, 외식, 골프장, 숙박, 면세점, 주유소 ...)",
        label_visibility="collapsed"
    )
    st.session_state["current_query"] = query
    st.caption("⌨️ 검색어를 입력하고 **엔터(Enter)**를 누르면 관련 차트 이미지가 아래에 즉시 나타납니다.")

    # 검색어 유무에 따른 분기
    if query.strip():
        with st.spinner(f"'{query}' 관련 차트 이미지를 탐색 중..."):
            res = searcher.search(query=query.strip(), top_k=top_k, alpha=alpha)
            raw_items = res.get("image_results", [])
            above_50 = sum(1 for it in raw_items if (it.get("final_score", 0.0) * 100) >= 50)
            below_50 = len(raw_items) - above_50
            top_score = max((it.get("final_score", 0.0) * 100 for it in raw_items), default=0)

        # 🎛️ 컴팩트 인라인 컨트롤 바 (검색창 바로 밑에 1줄 배치)
        bar_col1, bar_col2, bar_col3 = st.columns([1.5, 1.2, 0.8])
        with bar_col1:
            st.markdown(f"#### 🔍 **'{query}'** 검색 결과")
            st.caption(f"최고 일치율: **{top_score:.1f}%** | 50% 이상: **{above_50}장** | 50% 미만: **{below_50}장**")
        with bar_col2:
            min_sim = st.slider(
                "최소 일치율 필터 (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help="슬라이더를 좌우로 움직이면 해당 퍼센트 이상의 이미지가 실시간 반영됩니다."
            )
        with bar_col3:
            st.write("")  # 수직 정렬 공백
            show_below_50 = st.checkbox("50% 미만 보기", value=False, help="체크하면 50% 미만으로 제외된 저유사도 차트들을 확인합니다.")

        # 필터링 적용
        if show_below_50:
            items_to_display = [it for it in raw_items if (it.get("final_score", 0.0) * 100) < 50]
            st.info(f"📋 일치율 50% 미만으로 제외된 차트 목록입니다 ({len(items_to_display)}장).")
        else:
            items_to_display = [it for it in raw_items if (it.get("final_score", 0.0) * 100) >= min_sim]

        if not items_to_display:
            st.warning(f"🔍 현재 일치율 기준(<b>{min_sim}% 이상</b>)에 해당하는 차트 이미지가 없습니다.", icon="⚠️")
            st.info("💡 위의 슬라이더 핸들을 왼쪽으로 내려 기준을 낮추거나, '50% 미만 보기'를 체크해 보세요.")
            return
    else:
        # 검색어가 없을 때는 최신 주요 차트 피드 표시
        items_to_display = searcher.pages[:top_k]
        st.markdown(f"#### 📌 최신 제주 소비 트렌드 차트 피드 ({len(items_to_display)}장)")

    if not items_to_display:
        st.info("표시할 이미지가 없습니다.")
        return

    # 반응형 카드 그리드 렌더링 (Pinterest 스타일)
    cols = st.columns(grid_cols)
    for idx, item in enumerate(items_to_display):
        col = cols[idx % grid_cols]
        with col:
            img_path = resolve_image_path(item)
            if img_path:
                st.image(img_path, use_container_width=True)

            score_html = ""
            if "final_score" in item:
                score_val = int(item["final_score"] * 100)
                chip_class = "score-chip-high" if score_val >= 80 else "score-chip"
                score_html = f"<span class='{chip_class}'>일치율 {score_val}%</span>"

            chart_heading = get_chart_title(item)
            snippet_preview = item.get("snippet", "")
            if not snippet_preview and item.get("text"):
                clean_first_text = " ".join(item.get("text", "").split())
                snippet_preview = clean_first_text[:75] + ("..." if len(clean_first_text) > 75 else "")
            elif len(snippet_preview) > 75:
                snippet_preview = snippet_preview[:75] + "..."

            match_reason_text = get_match_reason(item, query) if query.strip() else "최신 정기 리포트 주요 시각화 차트"

            safe_heading = html.escape(chart_heading)
            safe_source = html.escape(item.get("report_title", ""))
            safe_reason = html.escape(match_reason_text)
            highlighted_snippet = highlight_text(snippet_preview, query) if snippet_preview else "본문 세부 분석 내용 수록"

            card_html = (
                f'<div class="card-meta">'
                f'<div class="card-title" title="{safe_heading}">{safe_heading}</div>'
                f'<div style="font-size:0.75rem; color:#718096; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
                f'출처: {safe_source}'
                f'</div>'
                f'<div class="card-sub">'
                f'<span>{item.get("report_date", "")} (p.{item.get("page_number", 1)})</span>'
                f'{score_html}'
                f'</div>'
                f'<div style="font-size:0.78rem; color:#2B6CB0; background:#EBF8FF; border-left:3px solid #3182CE; border-radius:4px; padding:6px 8px; margin-top:6px; line-height:1.4;">'
                f'💡 <b>일치 이유:</b> {safe_reason}'
                f'</div>'
                f'<div style="font-size:0.76rem; color:#4A5568; background:#F7FAFC; border:1px solid #E2E8F0; border-radius:4px; padding:5px 8px; margin-top:5px; line-height:1.35;">'
                f'📝 <b>본문 언급:</b> {highlighted_snippet}'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            # 상세 보기 모달 팝업 트리거
            if st.button("🔍 크게 보기 & 다운로드", key=f"view_btn_{idx}_{item.get('page_id')}", use_container_width=True):
                show_image_modal(item)


if __name__ == "__main__":
    main()
