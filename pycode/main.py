from langchain.llms import OpenAI
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

api_key = os.getenv('OPEN_API_KEY')
llm = OpenAI(openai_api_key=api_key)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)

result = code_chain({
"language": "Python",
    "task": "return a list of number"
})

print(result)

