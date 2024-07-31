from comm_init import init_llm, LlmModel, print_to_console
import os
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console(resp)

## Basic App for Evaluation
#**Load the text file**
from langchain.document_loaders import TextLoader

loader = TextLoader("data/be-good-and-how-not-to-die.txt")
document = loader.load()

#**The document is loaded as a Python list with metadata**
print_to_console(type(document))
print_to_console(len(document))
print_to_console(document[0].metadata)
print_to_console(f"You have {len(document)} document.")
print_to_console(f"Your document has {len(document[0].page_content)} characters")

#**Split the document in small chunks**
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400
)
document_chunks = text_splitter.split_documents(document)
print_to_console(f"Now you have {len(document_chunks)} chunks.")


from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
gemini_api_key = os.environ["gemini_api_key"]
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                             google_api_key=gemini_api_key)

#**Load the embeddings to a vector database**
from langchain_chroma import Chroma
from chromadb.config import Settings
stored_embeddings = Chroma.from_documents(document_chunks, embeddings, 
                                          persist_directory="./chroma_db",
                                          client_settings= Settings( anonymized_telemetry=False, 
                                                                    is_persistent=True, ))

from langchain.chains import RetrievalQA
QA_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=stored_embeddings.as_retriever(),
    input_key="question"
)

# Notice that we have added input_key in the QA_chain configuration. 
# This tells the chain where will the user prompt be located.

# **We are going to evaluate this app with 2 questions and 
# answers we already know (these answers are technically known as "ground truth answers")**

questions_and_answers = [
    {
        'question' : "Where is a whole neighborhood of YC-funded startups?", 
        'answer' :"In San Francisco"},
    {
        'question' : "What may be the most valuable  thing Paul Buchheit made for Google?", 
        'answer' : "The motto Don't be evil"}
]

predictions = QA_chain.apply(questions_and_answers)
# **The evaluation of this App has been positive, since the App has responded the 2 evaluation questions right.**

#**But instead of confirming that manually ourselves, we can ask the LLM to check if the responses are 
# coincidental with the "ground truth answers"**

from langchain.evaluation.qa import QAEvalChain

evaluation_chain = QAEvalChain.from_llm(llm)

evaluate_responses = evaluation_chain.evaluate(
    questions_and_answers,
    predictions,
    question_key="question",
    answer_key="answer"
)

print_to_console(evaluate_responses)