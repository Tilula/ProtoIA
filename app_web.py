import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Configurações
os.environ["TAVILY_API_KEY"] = "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"
GROQ_KEY = "gsk_HnOeR45..."

st.set_page_config(page_title="Proto IA | Fenix System", page_icon="🤖")
st.title("Proto IA - Fenix System")

# Inicialização do Modelo e Memória (Mantendo o estado no Streamlit)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=GROQ_KEY)
search_tool = TavilySearchResults(max_results=5)
tools = [search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é a Proto IA. Hoje é 23 de dezembro de 2025. Sua missão é fornecer respostas precisas."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
proto_ia_engine = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, handle_parsing_errors=True)

# Interface do Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Digite sua mensagem:"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        response = proto_ia_engine.invoke({"input": prompt_input})
        full_response = response['output']
        st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
