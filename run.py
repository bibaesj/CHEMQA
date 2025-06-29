import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import torch

# 1. JSON 데이터 로드
@st.cache_data
def load_data():
    with open("QA_EXPORT.json", "r", encoding="utf-8") as f:
        return json.load(f)

qa_data = load_data()

# 2. KoSBERT 모델 로드
@st.cache_resource
def load_model():
    return SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

model = load_model()

# 3. 질문 벡터화
corpus_questions = [entry["질문"] for entry in qa_data]
corpus_embeddings = model.encode(corpus_questions, convert_to_tensor=True)

# 4. 유사 질문 찾기
def find_best_match(user_input, top_k=1, threshold=0.4):
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(similarities, k=top_k)

    best_score = top_results.values[0].item()
    best_idx = top_results.indices[0].item()

    if best_score >= threshold:
        return qa_data[best_idx], best_score
    else:
        return None, best_score

# 5. Streamlit UI
st.title("🧪 CHEM-QA: 화생방 장비 고장 응답 시스템")
st.markdown("현재는 **원거리화학자동경보기**에 한정된 토이 프로젝트입니다.")

user_input = st.text_input("질문을 입력하세요", placeholder="예: 전원이 안 들어와요")

if user_input:
    result, score = find_best_match(user_input)
    if result:
        st.success(f"🔍 유사도: {score:.2f}")
        st.subheader("✅ 가능한 원인")
        for cause in result["원인"]:
            st.write(f"- {cause}")
        st.subheader("🔧 조치 사항")
        for action in result["조치사항"]:
            st.write(f"- {action}")
    else:
        st.warning("죄송합니다. 관련된 고장 정보를 찾지 못했습니다.")
