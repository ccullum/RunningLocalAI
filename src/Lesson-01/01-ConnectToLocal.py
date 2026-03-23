from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv 
 
# Load environment variables 
load_dotenv() 
 
#setup Engine
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="mistralai/ministral-3-3b",
    #model="microsoft/phi-4",
    #model="google/gemma-2-9b", 
    temperature=0.1
)
 
# Create a prompt template 
template = """Question: {question}  
Answer: Let's have a quick answer.""" 
 
prompt = ChatPromptTemplate.from_messages([
    ("system", " "),  # Here is where you define the system role!  Keep this as " " to enable for testing
    ("user", "Question: {question} \nAnswer: Let's have a quick answer.")
])

# Build the chain using LCEL 
chain = prompt | llm | StrOutputParser() 
# Run the chain 
response = chain.invoke({"question": "How many continents are there?"}) 
print(response)