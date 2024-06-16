from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAI
import streamlit as st
import os

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("api_key")


question_template = """
your are a chatbot.your job is to correctly and concisely answer all the questions that have been asked.you need to 
answer these questions formally and nicely.
Chat_history: {chat_history}
Question: {question} 
Answer:"""

llm = OpenAI(model='gpt-3.5-turbo-instruct',
             temperature=0)

memory = ConversationBufferMemory(memory_key="chat_history",
                                  max_len=50,
                                  return_messages=True,
                                  )

more_questions="Y"
while more_questions=="Y":
    question = input('What is your question:')
    question_prompt_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=question_template
    )
    llm_chain = LLMChain(llm=llm, prompt=question_prompt_template, memory=memory)
    output = llm_chain.invoke({'question': question})['text']
    print(output)
    memory.save_context({"input": question}, {"output": output})
    more_questions=input("Do you want to continue(Y/N):")

print(memory.load_memory_variables({}))
    



