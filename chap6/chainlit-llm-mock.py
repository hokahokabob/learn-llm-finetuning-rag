# -*- coding: sjis -*-

import chainlit as cl

def llm_main(q):
    return "�킩��܂���B"

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="�������͂���").send()

@cl.on_message
async def on_message(input_message):
    ans = llm_main(input_message.content)
    await cl.Message(content=ans).send()
    
    
