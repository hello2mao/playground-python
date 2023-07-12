from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import os

os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(temperature=0)

text_splitter = CharacterTextSplitter()

with open("doc.txt") as f:
    doc = f.read()
texts = text_splitter.split_text(doc)
print(f"texts len: {len(texts)}")

docs = [Document(page_content=t) for t in texts[:3]]
print(f"docs: {docs}")

chain = load_summarize_chain(llm, chain_type="stuff")
chain.run(docs)
