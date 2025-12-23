import os
import streamlit as st
from PIL import Image
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# --- CONFIGURAÇÃO DE SEGURANÇA E CHAVES ---
# Em 2025, o ideal é usar st.secrets ou variáveis de ambiente no Render
GROQ_KEY = "GROQ_API_KEY"
TAVILY_KEY = "TAVILY_API_KEY"
os.environ["TAVILY_API_KEY"] = TAVILY_KEY

# Configuração da página Web
st.set_page_config(page_title="Proto IA | Fenix System", layout="centered")

# Estilização Neon
st.markdown("""
    <style>
    .stApp { background-color: black; color: #00ff99; }
    .stChatMessage { border: 1px solid #00ff99; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Proto IA | Fenix System")
st.caption("Status: Online | Data: 23 de Dezembro de 2025")

# --- INICIALIZAÇÃO DO MOTOR DA IA ---
@st.cache_resource
def init_agent():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=GROQ_KEY)
    search_tool = TavilySearchResults(max_results=5)
    tools = [search_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é a Proto IA. Hoje é 23 de dezembro de 2025. "
                   "Sua missão é fornecer respostas precisas e profissionais. "
                   "Sempre analise os dados da pesquisa antes de responder."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Memória persistente na sessão do navegador
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=st.session_state.memory, 
        verbose=True, 
        handle_parsing_errors=True
    )

proto_ia_engine = init_agent()

# --- HISTÓRICO DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ENTRADA DE USUÁRIO ---
if prompt_input := st.chat_input("Como posso ajudar, Comandante?"):
    # Adiciona mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Gera resposta da IA
    with st.chat_message("assistant"):
        try:
            with st.spinner("Proto IA processando..."):
                response = proto_ia_engine.invoke({"input": prompt_input})
                full_response = response['output']
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Erro no motor da IA: {e}")
