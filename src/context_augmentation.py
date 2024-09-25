import os 
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Load the Flan-T5 Small model and tokenizer

model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name,token=HF_TOKEN)
model = T5ForConditionalGeneration.from_pretrained(model_name,token=HF_TOKEN)
 
def generate_augmented_response(query, retriever1_response, retriever2_response):


    combined_input = (
        f"User query: {query}\n"
        f"Context 1: {retriever1_response}\n"
        f"Context 2: {retriever2_response}\n"
        "Provide a concatenated and formatter response."
    )
 
    # Tokenize the combined input
    input_ids = tokenizer(combined_input, return_tensors="pt").input_ids
 
    # Generate a response with the Flan-T5 model
    outputs = model.generate(input_ids, max_length=500, num_beams=6, early_stopping=True)
 
    # Decode and return the final output
    #final_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return outputs
 
# Example user query and context retriever responses

user_query = "What is the status of the base stations?"
retriever1_response = "The base stations are located in three different regions."
retriever2_response = "Two base stations reported fan faults in the north region."
 
# Generate the final response based on the query and context
final_output = generate_augmented_response(user_query, retriever1_response, retriever2_response)
 
# Print the final augmented response
print(final_output)

