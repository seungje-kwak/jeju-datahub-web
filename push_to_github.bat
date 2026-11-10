@echo off
chcp 65001 > nul
title 깃허브(GitHub) 저장소 푸시 도우미
echo ======================================================================
echo  🚀 제주데이터허브 웹 서비스 GitHub 업로드 및 배포 준비
echo ======================================================================
echo.
cd /d "%~dp0"

echo 1. Git 저장소 초기화 중...
git init
git add .
git commit -m "feat: 제주데이터허브 AI 멀티모달 CLIP 차트 검색 시스템 구축"
git branch -M main

echo.
echo ======================================================================
echo GitHub에서 생성한 본인 원격 저장소 URL을 입력해 주세요.
echo (예: https://github.com/사용자아이디/jeju-datahub-web.git)
echo ======================================================================
set /p REPO_URL="저장소 URL: "

if "%REPO_URL%"=="" (
    echo [경고] URL이 입력되지 않았습니다. 나중에 수동으로 등록하세요:
    echo   git remote add origin [본인_URL]
    echo   git push -u origin main
    pause
    exit /b
)

git remote remove origin >nul 2>&1
git remote add origin %REPO_URL%
echo.
echo 2. 원격 저장소로 코드 푸시 중...
git push -u origin main

echo.
echo ======================================================================
echo  GitHub 푸시가 완료되었습니다!
echo  이제 https://share.streamlit.io 에서 배포를 진행하세요.
echo ======================================================================
pause
