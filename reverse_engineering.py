import openai

# Set your OpenAI API key
api_key = "sk-proj-vTVFSGpSD0UzkY7eCQtYT3BlbkFJqAG7XcfMoN1j9IVE3pp9"
openai.api_key = api_key

# Define the prompt for code generation
prompt = """
Generate a Python web scraping script using BeautifulSoup to extract grant information from the ACLS website.
The script should:
- Use the requests library to get the HTML content of the page.
- Use BeautifulSoup to parse the HTML content.
- Find the class name or ID that contains the grant information.
- Extract the grant information and store it in a list.
- Print the extracted grant information.
"""

# Request code generation from Codex
response = openai.Completion.create(
  engine="text-codex",  # Specify the Codex engine
  prompt=prompt,
  max_tokens=500,        # Limit the number of tokens in the response
)

# Print the generated Python code
print(response.choices[0].text.strip())
