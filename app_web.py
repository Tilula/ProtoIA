import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# --- Configuração da IA (Lendo chaves de forma segura via st.secrets) ---

# O Streamlit Cloud injeta as chaves do seu painel 'Secrets'
# Se rodar localmente, precisa garantir que as chaves estejam em .streamlit/secrets.toml
try:
    GROQ_KEY = st.secrets["GROQ_KEY"]
    TAVILY_KEY = st.secrets["TAVILY_KEY"]
except KeyError as e:
    st.error(f"Erro: Chave secreta {e} não encontrada. Configure no painel do Streamlit Cloud ou no arquivo .streamlit/secrets.toml localmente.")
    st.stop()

# Injetando no ambiente para as ferramentas que buscam automaticamente
os.environ["TAVILY_API_KEY"] = TAVILY_KEY

# Configuração do motor LangChain (agora totalmente seguro)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=GROQ_KEY)
search_tool = TavilySearchResults(max_results=5)
tools = [search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é a Proto IA. Hoje é 23 de dezembro de 2025. "
               "Sua missão é fornecer respostas precisas, profissionais e baseadas em dados atualizados da internet. "
               "Você deve ser um assistente prestativo, inteligente e analítico."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Inicializa a memória da conversa na sessão do Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Cria o executor do agente (reutiliza a memória da sessão)
agent = create_openai_tools_agent(llm, tools, prompt)
proto_ia_engine = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=st.session_state.chat_history, # Usa a memória da sessão
    verbose=False, 
    handle_parsing_errors=True
)

# --- Configuração da Interface Web com Streamlit ---

st.set_page_config(page_title="Proto IA Fenix", layout="centered")

# Adiciona a imagem do OVNI como logotipo
st.image("fundo_alien.png", width=150)

st.title("Sistema Proto IA Fenix (2025)")
st.caption("🤖 Assistente de IA com acesso à internet via Groq/Tavily.")

# Exibe o histórico de mensagens
for message in st.session_state.chat_history.buffer_as_messages:
    if message.type == "human":
        with st.chat_message("user"):
            st.write(message.content)
    elif message.type == "ai":
        with st.chat_message("assistant"):
            st.write(message.content)

# Campo de entrada de chat (st.chat_input é a melhor forma)
user_prompt = st.chat_input("Pergunte algo à Proto IA...")

if user_prompt:
    # Mostra a mensagem do usuário imediatamente
    with st.chat_message("user"):
        st.write(user_prompt)
    
    # Processa a resposta da IA
    with st.chat_message("assistant"):
        with st.spinner("Proto IA pensando..."):
            response = proto_ia_engine.invoke({"input": user_prompt})
            st.write(response['output'])
