# -*- coding: sjis -*-

import chainlit as cl
import os

os.environ['OPENAI_API_KEY'] = 'sk-****'

from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def llm_main(q):
    m = HumanMessage(content=q)
    ans = llm([m])
    return ans.content

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="何か入力せよ").send()

@cl.on_message
async def on_message(input_message):
    ans = llm_main(input_message.content)
    await cl.Message(content=ans).send()
