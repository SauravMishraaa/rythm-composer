from transformers import pipeline

def generate_lyrics(prompt):
    """
    Generate lyrics using GPT-NEO model.
    
    Args:
        prompt (str): The initial text prompt to generate lyrics from
        
    Returns:
        str: Formatted generated lyrics
    """
    # Initialize text generation
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
    
    # Generate lyrics based on the prompt
    response = generator(prompt, max_length=50, temperature=0.7, do_sample=True)
    
    
    output = response[0]['generated_text']
    
    cleaned_output = output.replace("\n", " ")
    formatted_lyrics = f"♪ {cleaned_output} ♪"
    
    return formatted_lyrics