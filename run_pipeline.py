# -*- coding: utf-8 -*-
"""
run_pipeline.py - 제주데이터허브 수집 -> 파싱/렌더링 -> 멀티모달 인덱싱 원클릭 파이프라인
"""

import sys
import time
from collector import fetch_and_download_reports
from extractor import extract_pages_and_render_images
from searcher import JejuMultimodalSearcher


def main(max_reports: int = 10):
    start_time = time.time()
    print("=" * 65)
    print(" [1단계] 제주데이터허브 최근 정기 리포트 수집 및 다운로드")
    print("=" * 65)
    reports = fetch_and_download_reports(max_reports=max_reports)

    print("\n" + "=" * 65)
    print(" [2단계] PDF 보고서 페이지별 텍스트 및 차트/표 고해상도 렌더링 (150 DPI)")
    print("=" * 65)
    pages = extract_pages_and_render_images(dpi=150)

    print("\n" + "=" * 65)
    print(" [3단계] 한국어 Multilingual-CLIP 멀티모달 벡터 인덱싱")
    print("=" * 65)
    searcher = JejuMultimodalSearcher()
    searcher.build_embeddings()

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f" 파이프라인 완료! (소요 시간: {elapsed:.1f}초)")
    print(f" - 수집된 보고서: {len(reports)}편")
    print(f" - 추출 및 렌더링된 페이지: {len(pages)}쪽")
    print(" 이제 다음 명령으로 검색 웹 데모(Streamlit)를 실행할 수 있습니다:")
    print("   streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(count)
