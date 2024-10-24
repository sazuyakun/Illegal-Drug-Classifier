from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="Llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define the structure for the output
class DrugTextAnalyzer():
    def __init__(self):
        # Define response schemas for classification, slang, and decoded terms
        self.response_schemas = [
            ResponseSchema(name="classification", description="The classification of the text (positive, negative, coded)"),
            ResponseSchema(name="identified_slang", description="A list of any slang or drug-related terms identified"),
            ResponseSchema(name="decoded_terms", description="A dictionary mapping slang terms to their decoded drug meanings"),
        ]

        # Create the output parser
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)

        # Create a prompt template for drug trafficking classification
        template = """
        You are tasked with classifying and decoding potential drug-related messages. Carefully analyze the user input for both explicit and implicit drug references, including slang, abbreviations, emojis, and cryptic language.

        Given the user input, classify the text into one of the following categories:
        1. "Positive" - Messages that explicitly refer to drugs, drug paraphernalia, pricing, or delivery methods.
        2. "Negative" - Messages unrelated to drugs or illegal activities.
        3. "Coded" - Messages that use slang, emojis, or cryptic language to refer to drugs or drug sales.

        For "Positive" or "Coded" messages, identify any slang or drug-related terms, and provide the decoded meaning of these terms.

        Return the response in the following JSON format:
        {{
            "classification": "<positive, negative, coded>",
            "identified_slang": ["<slang_term1>", "<slang_term2>", "..."],
            "decoded_terms": {{
                "<slang_term1>": "<decoded_meaning1>",
                "<slang_term2>": "<decoded_meaning2>"
            }}
        }}

        User input: {user_input}

        {format_instructions}
        """

        self.prompt = PromptTemplate(
            input_variables=["user_input"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
            template=template
        )

        # Create the LLM chain
        self.chain = self.prompt | llm

    # Function to process user input and classify the text
    def process_input(self, user_input):
        processed_output = []
        user_input = user_input.split('.')
        for ui in user_input:
            # Run the chain
            output = self.chain.invoke(ui)
            # Parse the output
            parsed_output = self.output_parser.parse(output.content)
            processed_output.append(parsed_output)
        
        return processed_output
