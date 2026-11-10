# -*- coding: utf-8 -*-
"""
extractor.py - PDF 보고서 페이지별 텍스트 및 차트/표 고해상도 이미지 렌더링 추출 모듈
- PyMuPDF(fitz) 기반 PDF 페이지 순회 및 텍스트 청크 추출
- 통계 차트, 범례, 축 폰트가 온전하게 보존되도록 150 DPI 고해상도 PNG 렌더링
- 검색용 인덱스 pages_index.json 생성 (상대 경로 및 크로스 플랫폼 호환)
"""

import os
import json
import pymupdf  # PyMuPDF
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
META_PATH = os.path.join(DATA_DIR, "reports_meta.json")
PAGES_INDEX_PATH = os.path.join(DATA_DIR, "pages_index.json")


def extract_pages_and_render_images(dpi: int = 150) -> List[Dict]:
    """
    수집된 PDF 보고서들을 순회하며:
    1. 각 페이지의 본문 텍스트 추출
    2. 각 페이지를 고해상도 PNG(차트·그래프 온전 보존)로 렌더링 저장
    3. 검색용 pages_index.json 생성 (상대 경로 표준화)
    """
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"메타데이터 파일이 없습니다: {META_PATH}. 먼저 collector.py를 실행하세요.")

    with open(META_PATH, "r", encoding="utf-8") as f:
        reports_meta = json.load(f)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    all_pages = []

    print(f"[2/3] PDF 보고서 텍스트 추출 및 고해상도 페이지 렌더링 시작 (DPI={dpi})...")

    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)

    for r_idx, report in enumerate(reports_meta, 1):
        report_id = report.get("report_id")
        title = report.get("title", f"Report_{report_id}")
        created_at = report.get("created_at", "")
        pdf_filename = report.get("pdf_filename")

        # 상대 경로 기반 PDF 파일 경로 탐색
        pdf_path = None
        if pdf_filename:
            candidate = os.path.join(REPORTS_DIR, pdf_filename)
            if os.path.exists(candidate):
                pdf_path = candidate

        if not pdf_path and report.get("pdf_path") and os.path.exists(report["pdf_path"]):
            pdf_path = report["pdf_path"]

        if not pdf_path or not os.path.exists(pdf_path):
            print(f"  * [{r_idx}/{len(reports_meta)}] PDF 파일 없음 건너뜀: {title}")
            continue

        try:
            doc = pymupdf.open(pdf_path)
            total_pages = len(doc)
            print(f"  * [{r_idx}/{len(reports_meta)}] {title} (총 {total_pages}페이지 처리 중...)")

            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text("text").strip()

                # 고해상도 이미지 렌더링
                image_filename = f"report_{report_id}_p{page_num + 1}.png"
                image_path = os.path.join(IMAGES_DIR, image_filename)
                rel_image_path = f"data/images/{image_filename}"

                if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    pix.save(image_path)

                all_pages.append({
                    "page_id": f"{report_id}_{page_num + 1}",
                    "report_id": report_id,
                    "report_title": title,
                    "report_date": created_at,
                    "page_number": page_num + 1,
                    "total_pages": total_pages,
                    "text": page_text,
                    "image_filename": image_filename,
                    "image_path": rel_image_path,
                    "pdf_filename": pdf_filename or os.path.basename(pdf_path),
                    "pdf_path": f"data/reports/{pdf_filename}" if pdf_filename else None,
                    "file_info_id": report.get("file_info_id"),
                    "file_id": report.get("file_id")
                })
            doc.close()
        except Exception as e:
            print(f"  ! PDF 처리 실패 ({title}): {e}")

    with open(PAGES_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    print(f"-> 총 {len(all_pages)}개 페이지 텍스트 및 이미지 추출 완료: {PAGES_INDEX_PATH}")
    return all_pages


if __name__ == "__main__":
    extract_pages_and_render_images()
