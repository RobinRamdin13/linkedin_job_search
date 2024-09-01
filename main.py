import os 
import json
import yaml
import faiss
import requests
import pandas as pd

from tqdm import tqdm
from typing import List, Tuple
from pandas import DataFrame
from dotenv import load_dotenv
from os.path import join, isdir
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.documents import Document
from uuid import uuid4

# load the environmental variables 
load_dotenv()

# instantiate global variables 
base_url = 'https://api.scrapingdog.com/linkedinjobs'
interested_cols = ['job_position', 'job_location', 'company_name','job_description', 'job_apply_link']

def format_field(field:str)->str: 
    """Function to format the space in field with '%20'

    Args:
        field (str): original field search

    Returns:
        str: formatted field search
    """    
    return field.replace(' ', '%20')

def get_request(data_dict:dict, extraction:str, raw:str)->None: 
    """Function to extract the meta data from LinkedIn

    Args:
        data_dict (dict): dictionary containing parameters for API call
        extraction (str): folder path
        raw (str): file path
    """    

    # base_url = 'https://api.scrapingdog.com/linkedinjobs' # instantiate the base url
    pages = data_dict['page']
    content = []
    # iterate through each page
    for page in tqdm(range(1, pages+1), desc='Extracting Data From LinkedIn'):
        # update the page number
        data_dict['page'] = page
        response = requests.get(base_url, params=data_dict) # instantiate the response
        assert response.status_code == 200 # ensure successful get request
        content.extend(json.loads(response.content.decode('utf-8')))
    
    df_raw = pd.DataFrame.from_dict(content)
    raw_file = join(extraction, raw) # file name to save to csv
    df_raw.to_csv(raw_file, index=False)
    return

def get_job_description(jobid:List, data_dict:dict, extraction:str, description:str)->None:
    """Function to extract the job descriptions for specific job id

    Args:
        jobid (List): List of Jobid
        data_dict (dict): dictionary containing API credential
        extraction (str): folder path
        description (str): file name
    """    
    job_list = []
    temp_dict = {}
    temp_dict['api_key'] = data_dict['api_key']
    for item in tqdm(jobid, desc='Extracting Job Descriptions'):
        temp_dict['job_id'] = item
        response = requests.get(base_url, params=temp_dict)
        assert response.status_code == 200 
        job_list.extend(json.loads(response.content.decode('utf-8')))
    
    df_description = pd.DataFrame.from_dict(job_list)
    descript_file = join(extraction,description)
    df_description.to_csv(descript_file, index=False)
    return

def create_documents(df:DataFrame)-> Tuple[List, List]:
    """Function to create the documents for indexing

    Args:
        df (DataFrame): dataframe containing the job description

    Returns:
        Tuple[List, List]: first list contains the list of document objects 
                            second list contains the unique ids of each document
    """    
    doc_list = []
    meta_cols = [f for f in df.columns.tolist() if f != 'job_description']
    for i in range(len(df)):
        doc_list.append(Document(page_content = df.loc[i, 'job_description'],
                                  metadata = df.loc[i, meta_cols].to_dict()))
    uuids = [str(uuid4()) for _ in range(len(doc_list))]
    return  doc_list, uuids

def main(config_path: str, skills_path:str, extraction:str, raw:str='raw.csv', description:str='job_desc.csv')->None:
    """Function to run the main logic of idenityfing the most relevant jobs based on a user defined skill set
    for a user defined role in Singapore.

    Args:
        config_path (str): configuration file path
        skills_path (str): skill file path
        extraction (str): extraction folder path
        raw (str, optional): file name for raw extraction. Defaults to 'raw.csv'.
        description (str, optional): file name for job description extraction. Defaults to 'job_desc.csv'.
    """    
    # load the yaml file
    with open(config_path, 'r') as file:
        data_dict = yaml.load(file, Loader=yaml.SafeLoader)
    with open(skills_path, 'r') as file1: 
        skills_dict = yaml.load(file1, Loader=yaml.SafeLoader)
    
    # format the field
    data_dict['field'] = format_field(data_dict['field'])

    # include the api key in data dict
    data_dict['api_key'] = os.getenv('SCRAPPING_DOG_API')
    
    # get the data from linkedin
    # get_request(data_dict, extraction, raw)

    # get the job description
    df_extraction = pd.read_csv(join(extraction, raw))
    # get_job_description(df_extraction['job_id'].values.tolist()[:15], data_dict, extraction, description)

    df_job = pd.read_csv(join(extraction, description), usecols=interested_cols)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},)
    
    document_list, uuids = create_documents(df_job)
    vector_store.add_documents(documents=document_list, ids=uuids)
    
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k':10})
    results = retriever.invoke(f"What is the best role for a {data_dict['field'].replace('%20', ' ')} with the following skills: {skills_dict['skills']}")
    result_list = []
    for result in results:
        result_list.append(result.metadata)
    df_output = pd.DataFrame.from_dict(result_list)
    print(df_output)
    return


if __name__ =='__main__':
    config_file = 'config.yaml'
    skill_file = 'skills.yaml'
    cwd = os.getcwd() # get the working directory
    extraction = join(cwd, 'extraction/') # create the extraction folder
    if not isdir(extraction): os.mkdir(extraction)
    main(config_file, skill_file, extraction)