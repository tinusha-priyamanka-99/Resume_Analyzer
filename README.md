## Overview
This project is a Generative AI tool designed to streamline the resume screening process for HR professionals. Leveraging the LangChain framework, OpenAI API, and Pinecone vector store, this application analyzes resumes, matches them to a provided job description, and provides a matching score along with a summary for each relevant resume. This allows HR teams to quickly identify the most suitable candidates.

## Features
- Upload Resumes: Users can upload multiple PDF resumes.
- Job Description Input: Users can input the job description against which resumes will be analyzed.
- Resume Matching: The tool compares each resume to the job description and returns the most relevant resumes.
- Summary Generation: Provides a summary of each relevant resume to facilitate quick review.
- Match Score: Assigns a score indicating how well each resume matches the job description.

## Code Explanation

  ### Part 1: Backend Logic
- Dependencies and Imports:<br>

  from langchain_openai import OpenAI<br>
  from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings<br>
  from langchain.schema import Document<br>
  import pinecone<br>
  from pypdf import PdfReader<br>
  from langchain.chains.summarize import load_summarize_chain<br>
  from langchain_community.vectorstores.pinecone import Pinecone<br>
  import os<br>

### Functions

- get_pdf_text(pdf_doc): Extracts text from a PDF file.<br>
- create_docs(user_pdf_list, unique_id): Creates a list of Document objects from the uploaded PDFs.<br>
- create_embeddings_load_data(): Loads the sentence transformer embeddings.<br>
- push_to_pinecone(...): Pushes the documents to Pinecone vector store.<br>
- pull_from_pinecone(...): Pulls documents from the Pinecone vector store.<br>
- similar_docs(...): Searches for similar documents in Pinecone vector store.<br>
- get_summary(current_doc): Generates a summary of the document using OpenAI.<br>

