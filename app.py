import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader
import tiktoken  # Import necessário para embeddings
import os
import streamlit.components.v1 as components

# Certifique-se de que a chave da API foi carregada da variável de ambiente
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.error("A chave da API OpenAI não foi encontrada. Por favor, configure a variável de ambiente 'OPENAI_API_KEY'.")
    st.stop()

# Carregar documentos
loader = CSVLoader(file_path="iso_base.csv")
documents = loader.load()

# Criar embeddings com a chave de API
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Indexar documentos no FAISS
db = FAISS.from_documents(documents, embeddings)

# Função para buscar informações
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

# Configuração do modelo de linguagem
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Template para o prompt
template = """
Responda à seguinte pergunta sobre as normas ISO de forma clara e acessível, fornecendo insights qualitativos e quantitativos: {message} 

Elabore uma análise concisa, destacando:
- Uma explicação clara da questão apresentada, incluindo causas, efeitos, estatísticas relevantes e dados de suporte.
- Avaliação crítica dos impactos e potenciais riscos associados.
- Estratégias recomendadas para mitigação.

Imagine sua audiência como executivos seniores e partes interessadas que precisam de uma compreensão abrangente do tema. 

Produza uma análise estruturada, livre de jargões técnicos, focada em resultados tangíveis e recomendações práticas.

Caso tenha dúvidas, aqui está uma lista de como responder a determinadas perguntas sobre as normas ISO: {best_practice}
"""

# Configurar o template do prompt
prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

# Configurar a cadeia de processamento
chain = LLMChain(llm=llm, prompt=prompt)

# Função para gerar resposta
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# Interface Streamlit

st.set_page_config(
    page_title="ISO Insight",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

components.html(
    """<!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Container Centralizado</title>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
       .container {
            text-align: center;
            margin: 10px auto; /* Reduce margin to 10px */
            padding: 20px;
            border: none;
            border-radius: 8px;
            width: 100%; /* Make the container width 100% of the column */
            background-color: #f0f0f0; /* Match the background color of the text input */
        }
       .container h1 {
            color: #333;
            margin-bottom: 10px;
        }
       .container p {
            color: #666;
            line-height: 1.6;
        }
        </style>
        </head>
        <body>

        <div class="container">
            <h1>ISO INSIGHT</h1>
            <p>Respostas rápidas e precisas para todas as suas dúvidas sobre normas ISO!</p>
        </div>

        </body>
        </html>"""
)

message = st.text_input("Digite sua pergunta:")
if st.button("Gerar análise"):
    if message:
        response = generate_response(message)
        st.subheader("Análise gerada!")
        st.write(response)
    else:
        st.warning("Por favor, insira uma pergunta.")

components.html(
    """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Footer Example</title>

        <style>
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

            * {
                font-family: 'Montserrat', sans-serif;
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                list-style: none;
                text-decoration: none;
            }

            .copyrights {
                padding: 20px;
                text-align: center;
                color: #000;
                background: #fff;
            }
        </style>
    </head>
    <body>
        <div class="copyrights">
            <p>&#169; Proteção Fácil All Rights Reserved</p>
        </div>
    </body>
    </html>
    """
)
