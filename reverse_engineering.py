import logging
import sys
import torch
from transformers import BitsAndBytesConfig
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index.readers import TrafilaturaWebReader
from llama_index import VectorStoreIndex, ServiceContext, SummaryIndex

root_url = "https://www.acls.org/fellows-grantees/?_fellow_year=2023&_paged="
# Update the range to include URLs for 5 pages
urls = [root_url + str(i) for i in range(1, 6)]

for page in range(1, 6):
    print("Loading data for page", page)
    documents = TrafilaturaWebReader().load_data([root_url + str(page)])
    print("Loaded", len(documents), "documents for page", page)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    def messages_to_prompt(messages):
        prompt = "Given the ACLS website URL: https://www.acls.org/fellows-grantees/?_fellow_year=2023&_paged=1, write a Python web scraper using BeautifulSoup to extract the grant information. Your scraper should:\n"
        prompt += "- Use the requests library to get the HTML content of the page.\n"
        prompt += "- Use BeautifulSoup to parse the HTML content.\n"
        prompt += "- Find the class name or ID that contains the grant information.\n"
        prompt += "- Extract the grant information and store it in a list.\n"
        prompt += "- Print the extracted grant information."
        return prompt

    llm = HuggingFaceLLM(
        model_name="HuggingFaceH4/zephyr-7b-alpha",
        tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
        query_wrapper_prompt=PromptTemplate("\n</s>\n\n{query_str}</s>\n\n"),
        context_window=3000,
        max_new_tokens=1600,
        model_kwargs={"quantization_config": quantization_config},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "do_sample": True},
        messages_to_prompt=messages_to_prompt,
        device_map="auto",
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model="local:BAAI/bge-large-en",
        chunk_size=512,
        chunk_overlap=64,
    )

    vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    summary_index = SummaryIndex.from_documents(documents, service_context=service_context)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    query_engine = vector_index.as_query_engine(response_mode="tree_summarize", similarity_top_k=4)

    query = "Extract the grant information and store it in a list."
    response = query_engine.query(query)
    print(query)
    print(response)

    # Clear memory
    del documents, quantization_config, llm, service_context, vector_index, summary_index, query_engine, response
    torch.cuda.empty_cache()

