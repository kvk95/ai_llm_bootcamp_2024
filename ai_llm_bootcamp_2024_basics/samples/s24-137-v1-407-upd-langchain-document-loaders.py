from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)

#region intro
# from datetime import datetime
# today = datetime.now().strftime("%dth%B")
# sprompt = f"""Who are some great personlites born on {today} in India?"""
# print(sprompt)
# llm_response = llm.invoke(sprompt)
# #TODO: uncomment
# print_to_console(llm_response)
#end region intro

## Chains
# Chains are sequences of operations. Usually, a chain combines:
# * a language model (LLM or Chat Model)
# * a prompt
# * other components

### Simple chain with LLMChain

## Document Loaders
# LangChain can load data from many sources:
# * Websites.
# * Databases.
# * Youtube, Twitter.
# * Excel, Pandas, Notion, Figma, HuggingFace, Github, Etc.

# LangChain can load data of many types:
# * PDF.
# * HTML.
# * JSON.
# * Word, Powerpoint, etc.

# **Sometimes you will have to clean or prepare the data you load before you can use it.**
# <br>
# This is something Data Scientist are used to do.

# region Loading PDF documents

# pip install pypdf

# from langchain.document_loaders import PyPDFLoader
# loader = PyPDFLoader("data/5pages.pdf")
# pages = loader.load()
# print_to_console(len(pages))
# page = pages[0]
# print_to_console(page)
# # #TODO: uncomment
# print_to_console(page.page_content[0:500])
# print_to_console(page.metadata)

# endregion Loading PDF documents

# region Loading YouTube Audio
# import os
# from yt_dlp import YoutubeDL
# from faster_whisper import WhisperModel


# url = "https://www.youtube.com/watch?v=Rb9Bpw8yvTg"
# save_dir = "data/youtube/"
# audio_file = os.path.join(save_dir, "audio4.mp3")

# os.makedirs(save_dir, exist_ok=True)
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# def download_audio(url, save_path):
#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'outtmpl': save_path,
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'mp3',
#             'preferredquality': '192',
#         }],        
#         'ffmpeg_location':os.path.realpath('F:/Downloads/Gyan.FFmpeg_7.0.1/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe'),
#     }

#     with YoutubeDL(ydl_opts) as ydl:
#         ydl.download([url])

# download_audio(url, audio_file)

# def transcribe_audio(audio_path):
#     model = WhisperModel("medium")
#     result = model.transcribe(audio_path)
#     segments, info = model.transcribe(audio_path)
#     return segments, info

# segments, info = transcribe_audio("data/youtube/audio4.mp3")
# print(segments)
# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# endregion Loading YouTube Audio

# region Loading websites

#**Option 1: Web Base Loader**
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://aiaccelera.com/100-ai-startups-100-llm-apps-that-have-earned-500000-before-their-first-year-of-existence/")
docs = loader.load()
print_to_console(docs[0].page_content[:2000])

#**Option 2: Unstructured HTML Loader**
# !pip install unstructured
from langchain.document_loaders import UnstructuredHTMLLoader
loader = UnstructuredHTMLLoader("data/_100 AI Startups__ 100 LLM Apps that have earned $500,000 before their first year of existence.html")
data = loader.load()
print_to_console(data)

#**Option 3: Beautiful Soup**
#!pip install beautifulsoup4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://aiaccelera.com/ai-consulting-for-businesses/")
data = loader.load()
print_to_console(data)

# endregion Loading websites