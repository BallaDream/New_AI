# main/rag_service.py

import pickle, faiss, os
# from langchain_community import OpenAIEmbeddings, ChatOpenAI # 최신 langchain-openai 사용
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from konlpy.tag import Okt
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.docstore.document import Document
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
import re
# ... (전처리 함수: clean_text, extract_pos, remove_stopwords 정의)

# API 키는 환경 변수에서 로드하거나, 안전한 방식으로 관리해야 합니다.
# settings.py에 API_KEY를 설정하고 os.environ.get('API_KEY')로 불러오는 것을 권장합니다.
OPENAI_API_KEY = "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" # 실제 키 입력
import re
from konlpy.tag import Okt

okt = Okt()

def clean_text(text):
    text = re.sub('[^가-힣a-zA-Z0-9.\s]', '', text)
    return re.sub(' +', ' ', text).strip()

def extract_pos(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    pos_tags = okt.pos(text, norm=True, stem=True)
    wanted_pos = ['Noun', 'Verb', 'Adjective']
    return ' '.join([word for word, pos in pos_tags if pos in wanted_pos])

def remove_stopwords(text):
    stopwords = ['하다','있다','같다','자다','않다','되다','쓰다','이다','진짜','써다','들다','되어다','너무','같아요','그래서','그리고']
    return ' '.join([w for w in text.split() if w not in stopwords])
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"#임시
def initialize_qa_chain():
    """RAG 체인을 서버 시작 시 단 한 번만 초기화하는 함수"""
    try:
        # 1) 로컬 파일 경로 설정 (manage.py 위치 기준)
        index_path = os.path.join("main", "rag_model", "cosmetic_faiss.index")
        data_path = os.path.join("main/rag_model","cosmetic_data2.pkl")
        if not os.path.exists(index_path):
            print("[DEBUG] S3에서 RAG 파일 다운로드 시작...")
            os.makedirs("main/rag_model/faiss_index", exist_ok=True) 
            s3 = boto3.client('s3')
            
            s3.download_file(S3_BUCKET_NAME, S3_FAISS_KEY, index_path)
            s3.download_file(S3_BUCKET_NAME, S3_PKL_KEY, data_path) # PKL 파일도 S3에 올려야 함
            print("[DEBUG] S3 다운로드 완료.")
        index = faiss.read_index("main/rag_model/faiss_index/cosmetic_faiss.index")
        
        with open("main/rag_model/faiss_index/cosmetic_data2.pkl", "rb") as f:
            data = pickle.load(f)
        texts = data["texts"]
        metadata = data["metadata"]
        docs = []
        for i in range(len(texts)):
            review_text = texts[i]
            meta = metadata[i]
            
            # 🌟 핵심 수정: metadata의 중요 정보를 review와 결합하여 page_content에 넣습니다.
            # 이렇게 해야 'acne', '필수' 같은 키워드가 임베딩 벡터에 반영됩니다.
            enhanced_content = (
                f"제품명: {meta.get('product_name', 'N/A')}. "
                f"피부고민_유형: {meta.get('type', 'N/A')}. "
                f"추천등급: {meta.get('grade', 'N/A')}. "
                f"리뷰: {review_text}"
            )

            doc = Document(page_content=enhanced_content, metadata=meta)
            docs.append(doc)
        # 2️⃣ Document로 변환
        # docs = [Document(page_content=texts[i], metadata=metadata[i]) for i in range(len(texts))]

        # 3️⃣ Docstore 구성
        docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})
        index_to_docstore_id = {i: str(i) for i in range(len(docs))}

        # 3) 벡터스토어 생성
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        # FAISS.from_embeddings 대신, 로드된 index를 사용하는 방식으로 수정
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 4) LLM + 프롬프트 세팅
        # template = """
        # 당신은 화장품 추천 전문가 AI입니다.
        # 아래는 사용자의 질문과 관련된 **리뷰 및 제품 정보(context)**입니다.
        # 이 정보(context)에 **포함된 내용만** 바탕으로 답변하세요.
        # 만약 관련 내용이 context에 없거나 불충분하면,
        # "해당 질문에 대한 정보가 데이터베이스에 없습니다."라고 답변하세요.

        # === 관련 리뷰 및 제품 데이터 ===
        # {context}

        # === 사용자 질문 ===
        # {question}

        # 위의 context에 근거하여, 
        # 피부 고민에 맞는 화장품을 추천하고 그 이유를 설명하세요.
        # """
        # template="""
        #     당신은 피부 전문가 AI 어시스턴트입니다. 당신의 목표는 사용자의 **피부 고민 유형(type)**과 **필요한 개선 등급(grade)**에 맞춰, 제공된 데이터베이스 정보만을 바탕으로 가장 적절한 화장품을 추천하는 것입니다.
        #     당신은 사용자 질문을 처리할 때, 다음의 **피부 고민 유형 매핑 규칙**을 알고 있습니다.
        #     - pigment: 색소침착
        #     - dry: 입술 건조 또는 건조
        #     - pore: 모공
        #     - wrinkle: 주름

        #     === 관련 리뷰 및 제품 데이터 (Context) ===
        #     {context}

        #     === 사용자 질문 ===
        #     {question}

        #     ---

        #     ### 답변 지침

        #     1.  **Context 집중**: 답변은 반드시 **{context}** 내에 포함된 **제품명, 피부고민 유형, 추천등급, 주요 성분, 리뷰** 정보를 근거로 합니다.
        #     2.  **등급 최우선**: 사용자 질문에 '필수', '권고', '예방'과 같은 **등급 정보가 포함**되어 있다면, Context 내에서 해당 등급에 맞는 제품을 최우선으로 추천해야 합니다.
        #     3.  **구체적인 이유**: 추천 화장품의 이름과 함께, Context 내의 **추천등급 및 주요 성분을 인용**하여 추천 이유를 구체적이고 전문적으로 설명하세요.
        #     4.  **정보 부족 시**: Context에 해당 질문에 대한 정보가 없으면, "죄송하지만, 현재 데이터베이스에는 해당 조건에 맞는 제품 정보가 충분하지 않습니다."라고 답변하세요.
        # """
        template = """
당신은 **피부 전문가 AI 어시스턴트**입니다. 당신의 목표는 제공된 데이터베이스 정보만을 바탕으로 사용자의 피부 고민과 필요한 개선 등급에 맞는 **가장 구체적인** 화장품을 추천하는 것입니다.
당신은 사용자 질문을 처리할 때, 다음의 **피부 고민 유형 매핑 규칙**을 알고 있습니다.
            - pigment: 색소침착
            - dry: 입술 건조 또는 건조
            - pore: 모공
            - wrinkle: 주름
            - elastic: 탄성
=== 관련 리뷰 및 제품 데이터 (Context) ===
{context}

=== 사용자 질문 ===
{question}

---

### 답변 지침 (반드시 다음 4가지 규칙을 최우선으로 따르세요)

1.  **Context 기반 답변 (유일한 근거)**: 답변은 반드시 **{context}** 내에 포함된 **제품명, 피부고민 유형, 추천등급, 주요 성분, 리뷰** 정보를 근거로 해야 합니다. Context를 벗어난 일반 지식이나 새로운 정보를 추가하지 마세요.

2.  **부위/유형 최우선**: 사용자 질문에 **특정 부위 또는 고민 유형** (예: 모공, 주름, 입술 건조, 색소침착)에 대한 언급이 있을 경우, Context 내에서 해당 유형이 명시된 제품을 최우선으로 검색하고 추천해야 합니다.

3.  **구체적인 추천 강제**: 추천하는 화장품의 이름과 함께, Context 내의 추천등급(필수/권고/예방), 그리고 리뷰 내용을 인용하여 추천 이유를 전문적으로 설명하세요.

4.  **정보 부족 시**: Context에 해당 질문에 대한 정보를 전혀 찾을 수 없거나 답변 근거가 불충분하면, **"죄송하지만, 현재 데이터베이스에는 해당 조건에 맞는 제품 정보가 충분하지 않습니다."**라고 답변하세요.
"""
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)

        # 5) QA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever, 
            chain_type="stuff", 
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    
    except Exception as e:
        print(f"RAG Chain 초기화 오류: {e}")
        return None
def preprocess_query(query):

    text = clean_text(query)

    text = extract_pos(text)

    text = remove_stopwords(text)

    return text
# 전역 변수로 체인 저장

# global_qa_chain = None
