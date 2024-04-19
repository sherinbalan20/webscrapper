import os
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
os.environ["TRANSFORMERS_CACHE"] = "/work/09959/pratyu5467/.cache/huggingface/"
os.environ["LLAMA_INDEX_CACHE_DIR"] = "/work/09959/pratyu5467/.cache/llama_index/"
# Define the prompt template
prompt_template = """
Given the ACLS website URL: https://www.acls.org/fellows-grantees/?_fellow_year=2023&_paged=1, 
write a Python web scraper using BeautifulSoup to extract the grant information. Your scraper should:
- Use the requests library to get the HTML content of the page.
- Use BeautifulSoup to parse the HTML content.
- Find the class name or ID that contains the grant information.
- Extract the grant information and store it in a list.
- Print the extracted grant information.
"""

# Define the LLM model and tokenizer
llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha", tokenizer_name="HuggingFaceH4/zephyr-7b-alpha")

# Define the prompt template
prompt_template = PromptTemplate(prompt_template)

# Generate the web scraping code using llm.complete
generated_code = llm.complete(prompt_template, max_tokens=100000)

# Print the generated code
print(generated_code)
