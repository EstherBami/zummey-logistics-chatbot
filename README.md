## Zummey Logistics Chatbot Project Documentation

### 1. **Project Overview**
The **Zummey Logistics Chatbot** is a Streamlit-based application designed to assist users with order delivery and logistics-related enquiries. The chatbot leverages:
- **LangChain** for natural language processing and retrieval-augmented generation (RAG).
- **Google Sheets** for storing and managing order details.
- **Groq API** for fast and efficient language model inference.
- **Hugging Face Embeddings** for creating a vector database of logistics-related documents.

The application allows users to:
1. **Place Orders**: Submit order details (sender, receiver, and instructions) which are saved to Google Sheets.
2. **Make Enquiries**: Ask logistics-related questions, with responses generated by the chatbot.


### **2. File Structure**
The project consists of the following files:

1. **`data_ingestion.py`**: Downloads a logistics-related PDF from Google Drive.
2. **`data_preprocessing.py`**: Processes the PDF to create a vector database using Hugging Face embeddings.
3. **`chatbot.py`**: Contains the Streamlit app, chatbot logic, and integration with Google Sheets.


### **3. Code Documentation**

#### **`data_ingestion.py`**
This script downloads a logistics-related PDF file from Google Drive and saves it to the `data` folder.

#### **`data_preprocessing.py`**
This script processes the downloaded PDF to create a vector database using Hugging Face embeddings and FAISS for efficient similarity search.

#### **`chatbot.py`**
This script contains the Streamlit app and chatbot logic. It integrates with Google Sheets for saving order details and uses the Groq API for generating responses.


### **4. Setup Instructions**

#### **Prerequisites**
1. Install Python 3.8+.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

#### **Environment Variables**
Create a `.env` file with the following variables:
```plaintext
GROQ_API_KEY=your_groq_api_key
CREDS_FILE_PATH=path/to/your/credentials.json
```

#### **Running the Project**
1. Download the logistics PDF:
   ```bash
   python data_ingestion.py
   ```
2. Create the vector database:
   ```bash
   python data_preprocessing.py
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run chatbot.py
   ```


### **5. Error Handling and Debugging**
- **Google Sheets Errors**: Ensure the worksheet name and spreadsheet ID are correct. Grant access to the service account email.
- **Groq API Errors**: Verify the API key is set in the `.env` file.
- **Vector Database Errors**: Ensure the PDF file is downloaded and processed correctly.


### **6. Future Improvements**
1. **User Authentication**: Add login functionality for secure access.
2. **Enhanced UI**: Improve the Streamlit UI with better styling and interactivity.

### **License**
This project is licensed under the MIT License. See the LICENSE file for details.

### **Contributing**
Contributions are welcome! Please open an issue or submit a pull request.