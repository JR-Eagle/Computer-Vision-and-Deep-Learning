import os
import requests
import textract
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain import PromptTemplate

# Set the OpenAI API key as an environment variable. 
# Replace '##########' with your actual key.
os.environ["OPENAI_API_KEY"] = '##########'

# Download the PDF from the internet
url = 'https://arxiv.org/pdf/2005.14165.pdf'
response = requests.get(url)
# Ensure that the request was successful
if response.status_code == 200:
    # Save the content of the response as a PDF file
    with open('LLM_Few-shot_learners.pdf', 'wb') as file:
        file.write(response.content)
else:
    # Print an error message if the PDF could not be downloaded
    print("Error: Unable to download the PDF file. Status code:",response.status_code)

# Convert the PDF to text using the textract library
doc = textract.process("LLM_Few-shot_learners.pdf")

# Save the converted text to a .txt file and then reopen it
# This helps to prevent potential encoding issues
with open('LLM_Few-shot_learners.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

# Reopen the saved .txt file and read its content into a variable
with open('LLM_Few-shot_learners.txt', 'r') as f:
    text = f.read()

# Assuming RecursiveCharacterTextSplitter is a class defined elsewhere that
# splits text into chunks based on character count
# Split the text into chunks with a specified size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # The size of each chunk of text
    chunk_overlap=24,  # The number of characters to overlap between chunks
    length_function=len,  # The function used to calculate the length of text
)

# Create chunks of documents from the text
chunks = text_splitter.create_documents([text])

# Initialize the embeddings object from the OpenAI library
embeddings = OpenAIEmbeddings()

# Create a vector database using FAISS to store and search document embeddings
db = FAISS.from_documents(chunks, embeddings)

# Define a query for the similarity search
query = "On which datasets does GPT-3 struggle?"

# Perform a similarity search in the FAISS database for the query
# 'k=5' returns the top 5 most similar documents
docs = db.similarity_search(query, k=5)

# Define a prompt template for formatting the input for the LLM
prompt = PromptTemplate(
    input_variables=["input"], 
    template="{input}",
)

# Format the prompt text using the defined template and the query
prompt_text = prompt.format(input=query)

# Load a QA chain which uses the OpenAI model to generate answers
chain = load_qa_chain(OpenAI(model_name="text-davinci-003", temperature=0.1, max_tokens=2000), chain_type="map_rerank")

# Run the QA chain with the input documents and the formatted question
chain.run(input_documents=docs, question=prompt_text)
