import os
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

# Set environment variables for Transformers and LLAMA_INDEX cache directories
os.environ["TRANSFORMERS_CACHE"] = "/work/09959/pratyu5467/.cache/huggingface/"
os.environ["LLAMA_INDEX_CACHE_DIR"] = "/work/09959/pratyu5467/.cache/llama_index/"

# Define the ACLS website URL
url = "https://www.acls.org/fellows-grantees/?_fellow_year=2023&_paged=1"

# Define the grant information
grant_info = """
Awarded individuals and their programs:

1. Mal Ahern
   - Program: ACLS Fellowship Program

2. Bimbola Akinbola
   - Program: Getty/ACLS Postdoctoral Fellowships in the History of Art

3. Sergio Alarcón Robledo
   - Program: Mellon/ACLS Dissertation Innovation Fellowships

...
"""

# Define the prompt template for generating Python web scraping script
prompt_template = f"""
Import the necessary libraries for web scraping:
- Use the requests library to get the HTML content of the page.
- Use BeautifulSoup to parse the HTML content.

Define the ACLS website URL:
url = "{url}"

Define the grant information:
grant_info = "{grant_info}"

Write a Python web scraper using BeautifulSoup to extract the grant information from the ACLS website:
- Get the HTML content of the page using requests.
- Parse the HTML content using BeautifulSoup.
- Find the class name or ID that contains the grant information.
- Extract the grant information and store it in a list.
- Print the extracted grant information.

"""

# Define the LLM model and tokenizer
llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha", tokenizer_name="HuggingFaceH4/zephyr-7b-alpha")

# Generate the Python web scraping script using llm.complete
generated_code = llm.complete(PromptTemplate(prompt_template), max_tokens=500)

# Print the generated Python code
print(generated_code)

