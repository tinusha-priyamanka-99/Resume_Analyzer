import streamlit as st
import uuid
import os
from utils import *
from dotenv import load_dotenv
load_dotenv()


if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

#def main():

st.set_page_config(page_title="Resume Screening Assistance")
st.title("HR - Resume Screening Assistance...")
st.subheader("I can help you in resume screening process")

job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...", key="1")
document_count = st.text_input("No.of 'RESUMES' to return", key="2")
pdf = st.file_uploader("Upload resumes here, only PDF files allowed",
                           type=["pdf"],
                           accept_multiple_files=True)
    
submit = st.button("Help me with the analysis")

if submit:
    with st.spinner('Wait for it...'):

        st.write("Our process")

        st.session_state['unique_id']=uuid.uuid4().hex
        #st.write(st.session_state['unique_id'])

        docs = create_docs(pdf,st.session_state['unique_id'])
        #st.write(docs)

        embeddings = create_embeddings_load_data()

        push_to_pinecone(
            os.environ.get("PINECONE_API_KEY"),
            "us-east-1",
            "resume-reader",
            embeddings,
            docs
        )

        relavant_docs = similar_docs(job_description,
                 document_count,
                 os.environ.get("PINECONE_API_KEY"),
                 "us-east-1",
                 "resume-reader",
                 embeddings,
                 st.session_state['unique_id'])
        
        #st.write(":heavy_minus_sign:" * 30)

        for item in range(len(relavant_docs)):

            st.subheader(str(item+1))

            st.write("**File** : "+ relavant_docs[item][0].metadata['name'])

            with st.expander('Show me'):
                st.info("**Match Score** : "+str(relavant_docs[item][1]))

                summary = get_summary(relavant_docs[item][0])
                st.write("**Summary** : "+summary )


        st.success("Hope I was able to save your time")


