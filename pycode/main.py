from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse
from dotenv import load_dotenv
import pprint

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--language', help='Programming language', default='Python')
parser.add_argument('--task', help='Task to be performed', default='return a list of number')
args = parser.parse_args()

# api_key = os.getenv('OPENAI_API_KEY')  by default it will take from environment variable
llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    template="Write a test function for the following {language} code:\n{code} ",
    input_variables=["language", "code"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"])
result = chain({"language": args.language, "task": args.task})

# print(f"Final Output\n:{result}")
# pretty print the result
pprint.pprint(result)
