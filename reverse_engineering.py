import os
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

os.environ["TRANSFORMERS_CACHE"] = "../.cache/huggingface/" # set cache dir for transformers
os.environ["LLAMA_INDEX_CACHE_DIR"] = "../.cache/llama_index/"

# Define the prompt template
prompt_template = """
Generate Python code to scrape grant information from the ACLS website for the year 2023.
Use BeautifulSoup to parse the HTML content.
Print the extracted grant information.
"""

# Define the LLM model and tokenizer
llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha", tokenizer_name="HuggingFaceH4/zephyr-7b-alpha")

# Define the prompt template
prompt_template = PromptTemplate(prompt_template)

# Generate the Python code using llm.complete
generated_code = llm.complete(prompt_template, max_tokens=2048)

# Print the generated code
print(generated_code)
