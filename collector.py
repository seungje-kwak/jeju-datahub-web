# -*- coding: utf-8 -*-
"""
collector.py - 제주데이터허브 정기 리포트 메타데이터 및 PDF 자동 수집 모듈
- 제주데이터허브 공식 REST API (/api/report/list) 연동
- 최신 정기 소비동향 분석 보고서 메타데이터 및 원본 PDF 일괄 다운로드
"""

import os
import re
import json
import urllib3
import requests
from typing import List, Dict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
META_PATH = os.path.join(DATA_DIR, "reports_meta.json")


def clean_filename(filename: str) -> str:
    """Windows 및 유닉스 환경에서 안전한 파일명으로 변환"""
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return safe_name.strip()


def fetch_and_download_reports(max_reports: int = 10) -> List[Dict]:
    """
    제주데이터허브 API에서 최근 정기 리포트 목록을 조회하고 PDF 파일을 다운로드합니다.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    list_url = (
        f"https://jejudatahub.net/api/report/list"
        f"?selection=title+%2B+contents&keyword=&isPaging=true&start=0&pageNumber=1&length={max_reports}&openYn=true"
    )

    print(f"[1/3] 제주데이터허브 API 목록 조회 중 (최대 {max_reports}건)...")
    try:
        resp = requests.get(list_url, verify=False, timeout=15)
        resp.raise_for_status()
        result_data = resp.json()
        records = result_data.get("data", [])
        print(f"-> 총 {len(records)}건의 리포트 메타데이터 수신 완료.")
    except Exception as e:
        print(f"-> API 조회 실패 또는 오프라인 모드: {e}")
        if os.path.exists(META_PATH):
            print(f"-> 기존 캐시된 메타데이터({META_PATH})를 유지합니다.")
            with open(META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    collected_list = []
    for idx, item in enumerate(records, 1):
        report_id = item.get("id")
        title = item.get("title", "").strip()
        contents_html = item.get("contents", "")
        created_at = item.get("createAt", "")
        file_obj = item.get("file") or {}
        file_infos = file_obj.get("fileInfos", [])

        pdf_info = None
        for fi in file_infos:
            if fi.get("fileName", "").lower().endswith(".pdf"):
                pdf_info = fi
                break

        if not pdf_info and file_infos:
            pdf_info = file_infos[0]

        local_pdf_path = None
        local_pdf_filename = None
        if pdf_info:
            file_id = pdf_info.get("fileId")
            file_info_id = pdf_info.get("id")
            original_file_name = pdf_info.get("fileName", f"report_{report_id}.pdf")
            raw_filename = f"{report_id}_{original_file_name}"
            local_pdf_filename = clean_filename(raw_filename)
            local_pdf_path = os.path.join(REPORTS_DIR, local_pdf_filename)

            download_url = f"https://jejudatahub.net/api/file?id={file_info_id}&fileId={file_id}"

            if not os.path.exists(local_pdf_path) or os.path.getsize(local_pdf_path) == 0:
                print(f"[{idx}/{len(records)}] 다운로드 중: {local_pdf_filename}")
                try:
                    dl_resp = requests.get(download_url, verify=False, timeout=30)
                    if dl_resp.status_code == 200:
                        with open(local_pdf_path, "wb") as f:
                            f.write(dl_resp.content)
                    else:
                        print(f"  * 다운로드 실패 (상태 코드: {dl_resp.status_code})")
                except Exception as dl_err:
                    print(f"  * 다운로드 오류: {dl_err}")
            else:
                print(f"[{idx}/{len(records)}] 이미 존재함: {local_pdf_filename}")

        collected_list.append({
            "report_id": report_id,
            "title": title,
            "contents_html": contents_html,
            "created_at": created_at,
            "pdf_filename": local_pdf_filename,
            "pdf_path": local_pdf_path,
            "file_info_id": pdf_info.get("id") if pdf_info else None,
            "file_id": pdf_info.get("fileId") if pdf_info else None,
            "original_filename": pdf_info.get("fileName") if pdf_info else None,
            "file_size": pdf_info.get("fileSize") if pdf_info else 0
        })

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(collected_list, f, ensure_ascii=False, indent=2)

    print(f"-> 메타데이터 저장 완료: {META_PATH}")
    return collected_list


if __name__ == "__main__":
    fetch_and_download_reports(10)
