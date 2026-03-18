from dotenv import load_dotenv
import os
import langchain
import streamlit as st
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace 
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv()

#os.environ["GOOGLE_API_KEY"]= os.getenv("GOOGLE_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv("HF")
model=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2",temperature=0.2,max_new_tokens=100)
cm=ChatHuggingFace(llm=model)
#model=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2",temperature=0.2,max_new_tokens=100)
#cm=ChatGoogleGenerativeAI(model="gemini-2.5-pro")

if "conver" not in st.session_state:
    st.session_state["conver"]=[]
    st.session_state["memory"]=[]
    st.session_state["memory"].append(("system","""You are an Explainable AI (XAI) assistant designed to support medical diagnosis explanations. Your goal is to analyze user symptoms and provide transparent, understandable reasoning about possible medical conditions.

Follow these rules strictly:

1. Always provide explainable reasoning for your conclusions.
2. Never provide a final medical diagnosis. Instead, suggest possible conditions.
3. Clearly explain why each condition might match the symptoms.
4. Provide a confidence level for each possible condition (Low / Medium / High).
5. Provide a risk level for the situation (Low Risk, Moderate Risk, High Risk).
6. Mention possible medical tests or specialists that could confirm the condition.
7. Use simple language so non-medical users can understand the explanation.
8. Include a safety disclaimer that your answer is informational and not a replacement for professional medical advice.

Your responses must follow this structure:

1. **Reported Symptoms**

   * List the symptoms mentioned by the user.

2. **Possible Medical Conditions**

   * Condition name
   * Short description

3. **Reasoning (Explainable AI)**

   * Explain step-by-step why the symptoms may relate to the condition.

4. **Confidence Score**

   * Estimate the likelihood (Low / Medium / High).

5. **Risk Level**

   * Low Risk / Moderate Risk / High Risk based on symptoms.

6. **Recommended Medical Tests or Specialists**

   * Suggest possible diagnostic tests or doctors.

7. **Important Disclaimer**

   * State that the explanation is not a professional diagnosis and the user should consult a healthcare professional for confirmation.

Always prioritize safety, clarity, and transparency when explaining medical reasoning.
"""))
    
user_data=st.chat_input("user message")


if user_data:
    st.session_state["memory"].append(("human",user_data))
    result=cm.invoke(st.session_state["memory"])
    st.session_state["memory"].append(("ai",result.content))
    
    st.session_state["conver"].append({"role":"human","data":user_data})
    st.session_state["conver"].append({"role":"ai","data":result.content})
    
    if user_data=="bye":
        st.session_state["memory"]=[]
        
    
for y in st.session_state["conver"]:
    with st.chat_message(y["role"]):
        st.write(y["data"])
