# 🚀 보험약관 챗봇 배포 가이드

## 1단계: GitHub 저장소 생성

1. https://github.com/new 접속
2. Repository name: `insurance-chatbot`
3. Description: `보험약관 Q&A 챗봇 - RAG 기반`
4. Public 선택
5. "Create repository" 클릭

## 2단계: GitHub에 푸시

```bash
git remote add origin https://github.com/MYUNGSOOK03/insurance-chatbot.git
git branch -M main
git push -u origin main
```

## 3단계: Streamlit Community Cloud 배포

1. https://share.streamlit.io/ 접속
2. "New app" 클릭
3. 설정:
   - Repository: `MYUNGSOOK03/insurance-chatbot`
   - Branch: `main`
   - Main file path: `app.py`
4. "Advanced settings" 클릭
5. Secrets 추가 (선택사항):
   ```toml
   # 사용자가 직접 입력하므로 필요 없음
   ```
6. "Deploy!" 클릭

## 4단계: URL 받기

배포 완료 후 받게 될 URL:
```
https://insurance-chatbot-[random].streamlit.app
```

## 5단계: 웹사이트에 링크 연결

`aimekkum-website/chatbot_project.html` 파일에서:

```html
<a href="https://insurance-chatbot-[받은URL].streamlit.app" target="_blank">
  🤖 챗봇 실행
</a>
```

## 배포 후 테스트

1. Streamlit 앱 URL 접속
2. OpenAI API Key 입력
3. 예시 질문으로 테스트
4. 웹사이트 "챗봇 실행" 버튼에서도 테스트

## 문제 해결

### PDF 파일이 너무 큼 (5.7MB)
- GitHub는 100MB까지 허용하므로 문제 없음
- Streamlit Cloud도 문제 없음

### API Key 관리
- 사용자가 직접 입력하므로 보안 문제 없음
- Streamlit Secrets에 저장할 필요 없음

## 비용
- GitHub: 무료
- Streamlit Cloud: 무료 (Community Cloud)
- OpenAI API: 사용자 부담

## 유지보수

### 업데이트 방법
```bash
git add .
git commit -m "Update chatbot"
git push
```
- Streamlit Cloud가 자동으로 재배포
