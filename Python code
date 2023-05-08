import pandas as pd
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import base64

# Function to load CSV
def load_csv(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    return loader.load(), pd.read_csv(tmp_file_path)

# Function to submit sidebar
def submit_sidebar(api_key, file):
    st.sidebar.empty()
    st.sidebar.success("API Key and CSV file submitted successfully.")

# Function to display data summary
def display_data_summary(df):
    st.write(f"Number of rows: {len(df)}")
    st.write("Key aspects of the data:")
    st.write(df.describe(include='all'))

# Function to display output
def display_output(output, container):
    if isinstance(output, pd.DataFrame):
        with container:
            st.dataframe(output)
    else:
        with container:
            message(output, key=str(len(st.session_state['generated'])), avatar_style="thumbs")

# Function to get rules from the UI
def get_rules():
    rules = []
    rule_input = st.sidebar.text_input("Enter rule (column_name, rule_condition, rule_value):")
    while rule_input:
        rules.append(tuple(rule_input.strip().split(",")))
        rule_input = st.sidebar.text_input("Enter rule (column_name, rule_condition, rule_value):")
    return rules


# Function to process CSV file
def process_csv_file(api_key, data,df):
    rules = get_rules() # get rules from the UI
    process_rules(ConversationalRetrievalChain.from_documents(data, OpenAIEmbeddings(openai_api_key=api_key)), rules, df=df)
    vectors = FAISS.from_documents(data, OpenAIEmbeddings(openai_api_key=api_key))
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=api_key),
        retriever=vectors.as_retriever()
    )


# Function to initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

# Function to handle conversational chat
def conversational_chat(chain, query, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# Function to display chat history
def display_chat_history(response_container):
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

# Function to process rules for GPT to make calculations for additional columns
def process_rules(chain, rules):
    # Process rules and modify the data accordingly
    # This function should be implemented based on the specific requirements and logic for processing the rules
    pass

# Function to download link for a DataFrame
def create_download_link(df, title="Download CSV file", filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{title}</a>'
    return href

# Initialize the sidebar elements
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your OpenAI API key, sk-",
    type="password"
)

uploaded_file = st.sidebar.file_uploader("Upload", type=["csv", "xls", "xlsx"])
submit_button = st.sidebar.button("Submit")

# Initialize the main UI elements
api_key_submitted = False
file_submitted = False
welcome_message = st.empty()

# Check if the submit button has been clicked
if submit_button:
    if user_api_key and uploaded_file:
        api_key_submitted = True
        file_submitted = True
        submit_sidebar(user_api_key, uploaded_file)
        welcome_message.empty()
    else:
        st.sidebar.warning("Please provide both API Key and CSV file.")

if api_key_submitted and file_submitted:
    try:
        data, df = load_csv(uploaded_file)
        st.success("Processing CSV file...")
        chain = process_csv_file(user_api_key, data)
        display_data_summary(df)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.warning("Please upload a valid CSV file to proceed.")
        st.stop()

    initialize_session_state()

    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk about your CSV data here :)", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            st.session_state['past'].append(user_input)
            output = conversational_chat(chain, user_input, st.session_state['history'])
            st.session_state['generated'].append(output)
            display_output(output, response_container)

    display_chat_history(response_container)

else:
    welcome_message.title("Welcome to CSV Analyzer")
    st.warning("Please submit your API Key and CSV file to proceed.")

