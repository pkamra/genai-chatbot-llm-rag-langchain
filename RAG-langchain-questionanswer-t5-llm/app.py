import yaml
import logging 
import requests
import json
import yaml
import boto3
from flask import Flask, request, jsonify
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from requests.auth import HTTPBasicAuth
# Your existing code here

app = Flask(__name__)

@app.route('/health')
def health_check():
    return '', 200

@app.route('/qa', methods=['POST'])
def question_answering():
    logger = logging.getLogger('sagemaker')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    
    sagemaker_client = boto3.client('runtime.sagemaker')
    # Get the question from the request
    question = request.json.get('question')
    logger.info(f'Question sent In :: {question}')

    # Your existing code to perform question answering
    #Now we need Langchain to first perform embedding of the question, find the answers, combine teh answers and then send to the LLM
    #https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/question_answering_retrieval_augmented_generation/question_answering_langchain_jumpstart.ipynb
    # Step 4. Use RAG based approach with LangChain and SageMaker endpoints to build a simplified question and answering application.

    
    TEXT_EMBEDDING_MODEL_ENDPOINT_NAME = 'jumpstart-dft-hf-textembedding-gpt-j-6b-fp16'
    # T5FLAN_XL_ENDPOINT_NAME = "jumpstart-example-huggingface-text2text-2023-08-05-07-33-26-290"
    T5FLAN_XXL_ENDPOINT_NAME = "jumpstart-example-huggingface-text2text-2023-08-06-16-40-45-080" 
    
    class SMLLMContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"
    
        def transform_input(self, prompt: str, model_kwargs: {}) -> bytes:
            input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
            return input_str.encode("utf-8")
    
        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            print (response_json)
            return response_json["generated_texts"][0]
    
    
    parameters = {
        "max_length": 1000, #restrict the length of the generated text
        "num_return_sequences": 1, # we will inspect several model outputs
        "top_k": 100, # top_k , Larger top_k values result in more conservative and focused text generation, while smaller values introduce more randomness and creativity.
        "top_p": 0.95, # top_p=0.8, for instance, the model will consider the minimum set of tokens that collectively account for at least 80% of the probability mass. This approach can generate diverse and coherent text while still maintaining control over the probability distribution
        "do_sample": False,
        "temperature": 1,
    }
    
    
    llm_content_handler = SMLLMContentHandler()
    
    sm_llm = SagemakerEndpoint(
        endpoint_name=T5FLAN_XXL_ENDPOINT_NAME,
        region_name="us-east-1",
        model_kwargs=parameters,
        content_handler=llm_content_handler,
    )
    
    with open('configsoa.yml', 'r') as file:
        config = yaml.safe_load(file)

    es_username = config['credentials']['username']
    es_password = config['credentials']['password']
    
    domain_endpoint = config['domain']['endpoint']
    domain_index = config['domain']['index']

    
    URL = f'{domain_endpoint}/{domain_index}/_search'
    logger.info(f'URL for OpenSearch index = {URL}')
    
    # question = 'What is the way to perform Cleaning and Disinfection during virus Outbreak?'
    # question ="Why should I select a  proper disinfectant ?"
    # question ="What is a PPE ?"
    # question = "What precautions should be taken for Cleaning and Disinfection of Equipment and Supplies?"
    # question = "What preventive measures should I take for Foot and Mouth disease?"
    
    
    #Next set of questions for washing machin mnaual
    # question = 'How should I clean the windshield?'
    # question = 'How should I check the tire inflation pressure?'
    # question = 'Where can I find the vehicle identification number?'
    # question = 'How can I report safety defects?'
    # question = 'What is a safety defect?'
    
    #get embedding from huggingface
    payload = {'text_inputs': [question]}
    payload = json.dumps(payload).encode('utf-8')
    response = sagemaker_client.invoke_endpoint(EndpointName=TEXT_EMBEDDING_MODEL_ENDPOINT_NAME, 
                                                ContentType='application/json', 
                                                Body=payload)
    body = json.loads(response['Body'].read())
    embedding = body['embedding'][0]
    
    #Now get responses for K-NNN nearest neighbor from ES
    K= 2
    
    query = {
        'size': K,
        'query': {
            'knn': {
              'embedding': {
                'vector': embedding,
                'k': K
              }
            }
          }
    }
    
    response = requests.post(URL, auth=HTTPBasicAuth(es_username, es_password), json=query)
    response_json = response.json()
    hits = response_json['hits']['hits']
    # print(hits.count)
    
    for hit in hits:
        # print(hit['_score'])
        # print(hit['_source']['passage'])
        score = hit['_score']
        passage = hit['_source']['passage']
        passage_id = hit['_source']['passage_id']
        # logger.info(f'Passage = {passage} | Score = {score} | PassageId = {passage_id}')
        
    if not hits:
        logger.warn('No matching documents found!')
        
    docs_page_content = " ".join([hit['_source']['passage'] for hit in hits])
    # print("combined context***********")
    # print(docs_page_content)
    print("starting langchain")
    
    
    # 1st Template to use for the system message prompt
    template = template = """
        You are a helpful assistant that can answer questions based on {docs}
    
    """
    
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    # Human question prompt
    # human_template = "Answer the following question: {question}.\\nSummarize\\n"
    human_template = "Answer the following question: {question}.\\nProvide detailed answer\\n"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    chain = LLMChain(llm=sm_llm, prompt=chat_prompt)
    
    response = chain.run(question=question, docs=docs_page_content)
    response = response.replace("\n", "")
    
        
    print("Response")
    print(response)
    
    
    
    #### 2nd template
    # class Document:
    #     def __init__(self, page_content, metadata=None):
    #         self.page_content = page_content
    #         self.metadata = metadata or {}
    
    
    # # Create a list of Document objects from the passages_list
    # docs = [Document(page_content=docs_page_content)]
    # print("Combined context")
    # print(docs[0].page_content)
    # prompt_template = """
    # Please answer the following question using the context provided. Be verbose and detailed
    
    # CONTEXT: 
    # {context}
    # =========
    # QUESTION: {question} 
    # ANSWER: """
    
    # PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # chain = load_qa_chain(
    #     llm=SagemakerEndpoint(
    #                 endpoint_name=T5FLAN_XXL_ENDPOINT_NAME,
    #                 region_name="us-east-1",
    #                 model_kwargs=parameters,
    #                 content_handler=llm_content_handler), 
    #     prompt=PROMPT)
    
    # result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    # print(result)

    return jsonify({"response": response})
    
    
if __name__ == '__main__':
    app.run()

