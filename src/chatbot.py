#from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """You are a helpful logistics assistant for Zummey Logistics. Your task is to answer user enquiries and collect order details.
If the information asked by the user is not available, politely inform them and suggest alternative solutions.
At the end of the conversation, provide a summary of the collected information on order delivery.

Context: {context}
User: {question}
Chatbot:

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for chatbot retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval Chatbot Chain
def retrieval_chatbot_chain(llm, prompt, db):
    chatbot_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return chatbot_chain

#Loading the model
def load_llm():

    load_dotenv() 
    KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="Llama-3.3-70b-Versatile", groq_api_key=KEY, temperature=0.8)

    return llm

#Chatbot Model Function
def chat_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True) 
    llm = load_llm()
    chatbot_prompt = set_custom_prompt()
    chatbot = retrieval_chatbot_chain(llm, chatbot_prompt, db)

    return chatbot

#output function
def final_result(query):
    qa_result = chat_bot()
    response = qa_result({'query': query})
    return response


# Google Sheets Configuration
SCOPE = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = os.getenv("CREDS_FILE_PATH")  # Get JSON key file path from environment variable
SHEET_NAME = "Order_Details" # Name of the work sheet
SPREADSHEET_ID = "1GI2nOijb1yHrZHlVbtZQK922CcxkttZztbxnRkd5DM0"  # Google Sheet ID

def save_order_to_sheets(order_details):
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SPREADSHEET_ID) # Open the spreadsheet using its ID
        worksheet = spreadsheet.worksheet(SHEET_NAME)  # Open the correct  work sheet

        # Check if headers exist, add them if they don't
        header_row = [key.replace('_', ' ').title() for key in order_details.keys()]
        if worksheet.row_count == 0:  # Check if the sheet is empty
            worksheet.append_row(header_row)

        # Append the order details
        order_values = list(order_details.values())
        worksheet.append_row(order_values)

        st.success("Order details saved to Google Sheets!")

    except Exception as e:
        st.error(f"Error saving to Google Sheets: {e}")

# Streamlit App
st.title("Zummey Logistics Chatbot")
st.subheader("Cheap and Easy Logistics")
st.write("Hi, Welcome to Zummey Logistics. I will be glad to assist with your order delivery.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User interaction: Place Order or Make Enquiry
user_choice = st.radio("What would you like to do?", ("Place Order", "Make Enquiry"))

#Form to place order
if user_choice == "Place Order":
    with st.form("order_details_form"):
        st.write("Please provide your order details:")
        sender_name = st.text_input("Your Name (Sender)")
        sender_phone = st.text_input("Your Phone Number")
        sender_email = st.text_input("Your Email Address")
        receiver_name = st.text_input("Receiver's Name")
        receiver_phone = st.text_input("Receiver's Phone Number")
        instructions = st.text_area("Specific Instructions")
        submitted = st.form_submit_button("Submit Order Details")

    # Prints order summary
    if submitted:
        order_details = {
            "sender_name": sender_name,
            "sender_phone": sender_phone,
            "sender_email": sender_email,
            "receiver_name": receiver_name,
            "receiver_phone": receiver_phone,
            "instructions": instructions
        }

        with st.chat_message("chatbot"):
            st.markdown("Thank you for providing your order details. Here's a summary for confirmation:")
            for key, value in order_details.items():
                st.write(f"- {key.replace('_', ' ').title()}: {value}")

            # Save order details to Google Sheets
            save_order_to_sheets(order_details) # Save to Google Sheets

            # ... (Your order confirmation logic)
            #st.session_state.messages.append({"role": "chatbot", "content": "Thank you for providing your order details."})

# For enquiry
elif user_choice == "Make Enquiry":
    user_enquiry = st.text_input("Enter your enquiry:")

    # Process user input and display response
    if user_enquiry:
        with st.chat_message("user"):
            st.markdown(user_enquiry)
        st.session_state.messages.append({"role": "user", "content": user_enquiry})

        response = final_result(query=user_enquiry)

        with st.chat_message("chatbot"):
            st.markdown(response['result'])
        st.session_state.messages.append({"role": "chatbot", "content": response['result']})
