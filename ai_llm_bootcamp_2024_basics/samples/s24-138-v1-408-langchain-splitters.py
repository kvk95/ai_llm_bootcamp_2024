from comm_init import init_llm, LlmModel, print_to_console
llm = init_llm(llmmodel=LlmModel.MISTRAL)
# #TODO: uncomment
# print_to_console('')

#region intro
# from datetime import datetime
# today = datetime.now().strftime("%dth%B")
# sprompt = f"""Who are some great personlites born on {today} in India?"""
# print(sprompt)
# llm_response = llm.invoke(sprompt)
# #TODO: uncomment
# print_to_console('')
#end region intro

## Splitters
# * We use splitters to divide documents in small chunks for the RAG technique.
# * The way we split one document is very relevant, since it has a big impact in the quality of the RAG retrieval.
# * It is important to understand how to optimize the splitting process.
# * Two important techniques to optimize splitting are:
#     * How we build the chunks (ideally, whole sentences or paragraphs).
#     * The metadata we add to each chunk.

# region Character Splitter
# * This splits based on characters (by default "\n\n") and measure chunk length by number of characters.

from langchain.text_splitter import CharacterTextSplitter
chunk_size =26
chunk_overlap = 4
character_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
text1 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
# split = character_splitter.split_text(text1)
# #TODO: uncomment
# print_to_console(split)

text2 = """
Data that Speak
LLM Applications are revolutionizing industries such as 
banking, healthcare, insurance, education, legal, tourism, 
construction, logistics, marketing, sales, customer service, 
and even public administration.

The aim of our programs is for students to learn how to 
create LLM Applications in the context of a business, 
which presents a set of challenges that are important 
to consider in advance.
"""
# split2 = character_splitter.split_text(text2)
# #TODO: uncomment
# print_to_console(split2)

# endregion Character Splitter

# region Recursive Character Splitter
# This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

from langchain.text_splitter import RecursiveCharacterTextSplitter
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
rec_split1 = recursive_splitter.split_text(text1)
# #TODO: uncomment
print_to_console(rec_split1)

rec_split2 = recursive_splitter.split_text(text2)
# #TODO: uncomment
print_to_console(rec_split2)

second_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

seco_rec_split1 = second_recursive_splitter.split_text(text2)
# #TODO: uncomment
print_to_console(seco_rec_split1)

chunks = second_recursive_splitter.split_text(text2)
# #TODO: uncomment
print_to_console(len(chunks))

# region Adding helpful metadata to the text 

from langchain.text_splitter import MarkdownHeaderTextSplitter
document_with_markdown = """
# Title: My book\n\n \

## Chapter 1: The day I was born\n\n \
I was born in a very sunny day of summer...\n\n \

### Section 1.1: My family \n\n \
My father had a big white car... \n\n 

## Chapter 2: My school\n\n \
My first day at the school was...\n\n \

"""
headers_to_split_on = [
    ("#", "Book title"),
    ("##", "Chapter"),
    ("###", "Section"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_header_splits = markdown_splitter.split_text(document_with_markdown)

print_to_console(md_header_splits)
print_to_console(md_header_splits)
print_to_console(md_header_splits)

# endregion Adding helpful metadata to the text chunks

# endregion Recursive Character Splitter