import os
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = "/work/09959/pratyu5467/.cache/huggingface/"
os.environ["LLAMA_INDEX_CACHE_DIR"] = "/work/09959/pratyu5467/.cache/llama_index/"

# Define the base URL for ACLS website
base_url = "https://www.acls.org/fellows-grantees/?_fellow_year=2023&_paged="

# Define the LLM model and tokenizer
llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha", tokenizer_name="HuggingFaceH4/zephyr-7b-alpha")

# Loop through all pages (replace 10 with the total number of pages)
for page_num in range(1, 3):
    # Define the prompt template with the current page number
    prompt_template = base_url + str(page_num)
    prompt_template = PromptTemplate(prompt_template)

    # Generate the web scraping code using llm.complete
    generated_code = llm.complete(prompt_template, max_tokens=500)

    # Print the generated code
    print(f"Generated code for page {page_num}:")
    print(generated_code)

