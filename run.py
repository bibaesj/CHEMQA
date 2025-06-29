import streamlit as st
import json
import difflib

# 데이터 불러오기
@st.cache_data
def load_data():
    with open("QA_EXPORT.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

qa_data = load_data()

# 유사 질문 찾기
def find_best_match(user_question, qa_list):
    questions = [entry["질문"] for entry in qa_list]
    matches = difflib.get_close_matches(user_question, questions, n=1, cutoff=0.4)
    if matches:
        for entry in qa_list:
            if entry["질문"] == matches[0]:
                return entry
    return None

# Streamlit UI
st.title("🧪 원거리화학자동경보기 QA 챗봇")
st.markdown("질문을 입력하면 고장 원인과 조치 사항을 알려드립니다.")

user_input = st.text_input("질문을 입력하세요", placeholder="예: '탐지기 제어기에 전원이 안 들어와요'")

if user_input:
    result = find_best_match(user_input, qa_data)
    if result:
        st.subheader("✅ 가능한 원인")
        for cause in result["원인"]:
            st.write(f"- {cause}")
        st.subheader("🔧 조치 사항")
        for action in result["조치사항"]:
            st.write(f"- {action}")
    else:
        st.warning("죄송합니다. 해당 질문에 대한 답변을 찾지 못했습니다.")
