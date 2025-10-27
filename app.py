"""
보험약관 챗봇 - RAG 기반 Q&A 시스템
"""
import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 페이지 설정
st.set_page_config(
    page_title="🤖 보험약관 챗봇",
    page_icon="🤖",
    layout="wide"
)

# 제목
st.title("🤖 보험약관 챗봇")
st.markdown("**다이렉트 웰빙 건강보험** 약관에 대해 질문하세요!")

# OpenAI API 키 입력
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.warning("⚠️ 왼쪽 사이드바에 OpenAI API Key를 입력해주세요.")
    st.info("API Key는 https://platform.openai.com/api-keys 에서 발급받을 수 있습니다.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# PDF 파일 경로 (Streamlit Cloud용 상대 경로)
import os
# PDF 파일 경로
PDF_FILE = "insurance_policy.pdf"

# 파일 존재 확인
if not os.path.exists(PDF_FILE):
    st.error(f"❌ PDF 파일을 찾을 수 없습니다: {PDF_FILE}")
    st.info(f"현재 디렉토리: {os.getcwd()}")
    st.info(f"디렉토리 내 파일: {os.listdir('.')}")
    st.stop()

@st.cache_resource
def load_pdf_and_create_vectorstore():
    """PDF를 로드하고 벡터 데이터베이스를 생성"""
    try:
        # PDF 파일 크기 확인
        file_size = os.path.getsize(PDF_FILE)
        st.info(f"📄 PDF 파일 크기: {file_size / 1024 / 1024:.2f} MB")
        
        if file_size == 0:
            st.error("❌ PDF 파일이 비어있습니다.")
            return None, 0
        
        # PDF 읽기
        pdf_reader = PdfReader(PDF_FILE)
        st.info(f"📖 PDF 페이지 수: {len(pdf_reader.pages)}")
        
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += page_text
            if i == 0:  # 첫 페이지 샘플 출력
                st.info(f"첫 페이지 텍스트 샘플: {page_text[:100]}...")
        
        if not text.strip():
            st.error("❌ PDF에서 텍스트를 추출할 수 없습니다.")
            return None, 0
        
        st.info(f"📝 추출된 텍스트 길이: {len(text)} 문자")
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.info(f"✂️ 청크 분할 완료: {len(chunks)}개")
        
        # 벡터 데이터베이스 생성
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        return vectorstore, len(chunks)
    except Exception as e:
        st.error(f"❌ PDF 로딩 중 오류: {str(e)}")
        import traceback
        st.error(f"상세 오류:\n{traceback.format_exc()}")
        return None, 0

# 벡터 데이터베이스 로드
with st.spinner("📄 보험약관 문서를 분석 중입니다..."):
    vectorstore, num_chunks = load_pdf_and_create_vectorstore()

if vectorstore:
    st.sidebar.success(f"✅ 문서 로딩 완료! ({num_chunks}개 청크)")
    
    # 채팅 히스토리 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # LLM 설정
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # 프롬프트 템플릿
    template = """다음 컨텍스트를 사용하여 질문에 답변하세요. 
    컨텍스트에서 답을 찾을 수 없으면, 모른다고 말하세요.
    
    컨텍스트: {context}
    
    질문: {question}
    
    답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # RAG 체인 생성
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 채팅 인터페이스
    st.markdown("---")
    
    # 예시 질문
    st.sidebar.markdown("### 💡 예시 질문")
    example_questions = [
        "입원비 청구에 필요한 서류는?",
        "보장 개시일은 언제인가요?",
        "암 진단비는 얼마나 받을 수 있나요?",
        "면책기간이 무엇인가요?",
        "보험료 납입은 어떻게 하나요?"
    ]
    
    for q in example_questions:
        if st.sidebar.button(q, key=f"example_{q}"):
            st.session_state.current_question = q
    
    # 채팅 기록 표시
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 참고 근거"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**[근거 {i}]**\n{source[:300]}...")
    
    # 질문 입력
    question = st.chat_input("보험약관에 대해 질문해주세요...")
    
    # 예시 질문 클릭 시
    if "current_question" in st.session_state:
        question = st.session_state.current_question
        del st.session_state.current_question
    
    if question:
        # 사용자 질문 표시
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # AI 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하고 있습니다..."):
                try:
                    # 관련 문서 검색
                    docs = retriever.invoke(question)
                    context = format_docs(docs)
                    
                    # LLM으로 답변 생성
                    chain = (
                        {"context": lambda x: context, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    answer = chain.invoke(question)
                    
                    st.markdown(answer)
                    
                    # 근거 표시
                    if docs:
                        sources = [doc.page_content for doc in docs]
                        with st.expander("📚 참고 근거"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**[근거 {i}]**\n{source[:300]}...")
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })
                except Exception as e:
                    st.error(f"답변 생성 중 오류: {str(e)}")
    
    # 대화 초기화 버튼
    if st.sidebar.button("🔄 대화 초기화"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.error("❌ 문서 로딩에 실패했습니다.")

# 사이드바 정보
st.sidebar.markdown("---")
st.sidebar.markdown("""
### ℹ️ 사용 방법
1. OpenAI API Key 입력
2. 보험약관에 대해 질문 입력
3. AI가 근거와 함께 답변 제공

### 🔧 기술 스택
- LangChain (RAG)
- FAISS (벡터 DB)
- OpenAI GPT-3.5
- Streamlit
""")
