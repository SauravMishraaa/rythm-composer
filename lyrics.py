from transformers import pipeline

def generate_lyrics(prompt):
    """
    Generate lyrics using GPT-NEO model.
    
    Args:
        prompt (str): The initial text prompt to generate lyrics from
        
    Returns:
        str: Formatted generated lyrics
    """
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
    
    response = generator(prompt, max_length=100, temperature=0.5, eos_token_id=None, top_p=0.9, do_sample=True)
    
    output = response[0]['generated_text']
    
    # cleaned_output = output.replace("\n", " ")
    formatted_lyrics = f"♪ {output} ♪"
    
    return formatted_lyrics