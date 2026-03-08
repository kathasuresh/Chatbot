from dotenv import load_dotenv
import os
import langchain
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

os.environ["GOOGLE_API_KEY"]= os.getenv("GOOGLE_API_KEY")

cm=ChatGoogleGenerativeAI(model="gemini-2.5-pro")

if "conver" not in st.session_state:
    st.session_state["conver"]=[]
    st.session_state["memory"]=[]
    st.session_state["memory"].append(("system","answer as a twenty year old boy"))
    
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
