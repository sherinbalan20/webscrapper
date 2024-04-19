import os
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

# Function to convert messages to prompt
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"\n{message.content}</s>\n"

    if not prompt.startswith("\n"):
        prompt = "\n</s>\n" + prompt
    prompt = prompt + "\n"

    return prompt

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = "/work/09959/pratyu5467/.cache/huggingface/"
os.environ["LLAMA_INDEX_CACHE_DIR"] = "/work/09959/pratyu5467/.cache/llama_index/"

# Define the base URL for ACLS website
base_url = "https://www.acls.org/fellows-grantees/?_fellow_year=2023&_paged="

# Define the LLM model and tokenizer
llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-alpha", tokenizer_name="HuggingFaceH4/zephyr-7b-alpha")

# Loop through all pages (replace 10 with the total number of pages)
for page_num in range(1, 11):  # Assuming there are 10 pages
    # Define the full URL for the current page
    page_url = base_url + str(page_num)

    # Define the messages to be used as the prompt
    messages = [
        {"role": "system", "content": "Given the ACLS website URL: https://www.acls.org/fellows-grantees/?_fellow_year=2023&_paged=1, write a Python web scraper using BeautifulSoup to extract the grant information. Your scraper should:"},
        {"role": "assistant", "content": "- Use the requests library to get the HTML content of the page."},
        {"role": "assistant", "content": "- Use BeautifulSoup to parse the HTML content."},
        {"role": "assistant", "content": "- Find the class name or ID that contains the grant information."},
        {"role": "assistant", "content": "- Extract the grant information and store it in a list."},
        {"role": "assistant", "content": "- Print the extracted grant information."}
    ]

    # Generate the prompt from messages
    prompt_template = messages_to_prompt(messages)

    try:
        # Generate the web scraping code using llm.complete
        generated_code = llm.complete(prompt_template, max_tokens=2000)  # Increase max_tokens as needed

        # Print the generated code
        print(f"Generated code for page {page_num}:")
        print(generated_code)
    except Exception as e:
        print(f"Error occurred while scraping page {page_num}: {e}")
