# 🤖 보험약관 챗봇

다이렉트 웰빙 건강보험 약관을 질문-답변 형태로 쉽게 이해할 수 있는 RAG 기반 챗봇입니다.

## 📋 기능

- ✅ PDF 약관 자동 분석
- ✅ 자연어 질문으로 약관 조회
- ✅ 근거와 함께 답변 제공
- ✅ 대화 히스토리 유지
- ✅ 예시 질문 제공

## 🚀 실행 방법

### 1. 가상환경 생성 및 활성화

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 챗봇 실행

```bash
streamlit run app.py
```

### 4. OpenAI API Key 입력

- 왼쪽 사이드바에 API Key 입력
- https://platform.openai.com/api-keys 에서 발급

## 💡 사용 예시

**질문 예시:**
- "입원비 청구에 필요한 서류는?"
- "보장 개시일은 언제인가요?"
- "암 진단비는 얼마나 받을 수 있나요?"
- "면책기간이 무엇인가요?"

## 🔧 기술 스택

- **LangChain**: RAG 파이프라인
- **FAISS**: 벡터 데이터베이스
- **OpenAI GPT-3.5**: LLM
- **Streamlit**: 웹 인터페이스
- **PyPDF2**: PDF 파싱

## 📁 파일 구조

```
insurance_chatbot/
├── app.py                              # 메인 애플리케이션
├── requirements.txt                    # 패키지 목록
├── msook다이렉트웰빙건강보험_약관.pdf # 보험약관 PDF
└── README.md                           # 문서
```

## 📝 작동 원리

1. **PDF 로딩**: PyPDF2로 보험약관 PDF 텍스트 추출
2. **청크 분할**: 1000자 단위로 텍스트 분할 (오버랩 200자)
3. **임베딩**: OpenAI Embeddings로 벡터화
4. **벡터 저장**: FAISS에 벡터 인덱스 저장
5. **질문 처리**: 사용자 질문을 벡터화하여 유사한 청크 검색
6. **답변 생성**: 검색된 청크를 컨텍스트로 GPT-3.5가 답변 생성

## ⚙️ 환경 변수

OpenAI API Key는 Streamlit 사이드바에서 입력합니다.

## 📞 문의

문제가 발생하면 이슈를 등록해주세요.
