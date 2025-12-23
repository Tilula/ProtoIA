import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Configuração de chaves
GROQ_KEY = os.environ.get("GROQ_KEY")
TAVILY_KEY = os.environ.get("TAVILY_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_KEY

# Configuração da IA
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=GROQ_KEY)
search_tool = TavilySearchResults(max_results=5)
tools = [search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é a Proto IA. Hoje é 23 de dezembro de 2025."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# O segredo está aqui: usamos create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
proto_ia_engine = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=False, handle_parsing_errors=True)

# Interface
st.set_page_config(page_title="Proto IA Fenix")
st.title("Sistema Proto IA Fenix")

for message in st.session_state.memory.buffer_as_messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message.content)

user_prompt = st.chat_input("Pergunte algo...")
if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)
    with st.chat_message("assistant"):
        response = proto_ia_engine.invoke({"input": user_prompt})
        st.write(response['output'])
