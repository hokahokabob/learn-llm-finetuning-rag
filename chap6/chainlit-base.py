# -*- coding: sjis -*-

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="�������͂���").send()

@cl.on_message
async def on_message(input_message):
    await cl.Message(content=(input_message.content + "�A�A�A�ł���")).send()
    
