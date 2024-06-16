from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAI
import chainlit as cl
import os

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("api_key")


question_template = """
your are a chatbot.your job is to correctly and concisely answer all the questions that have been asked.you need to 
answer these questions formally and nicely.
Chat_history: {chat_history}
Question: {question} 
Answer:"""

question_prompt_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=question_template
    )

@cl.on_chat_start
def query_llm():
    llm = OpenAI(model='gpt-3.5-turbo-instruct',
                temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history",
                                    max_len=50,
                                    return_messages=True,
                                    )
    llm_chain = LLMChain(llm=llm, prompt=question_prompt_template, memory=memory)
    
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    response = await llm_chain.acall(message.content, 
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(response["text"]).send()

    



