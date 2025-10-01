import re

def clean_text(text):
    """
    Clean and normalize license plate text
    
    Args:
        text (str): Raw text from OCR
        
    Returns:
        str: Cleaned license plate text or empty string if invalid
    """
    if not text:
        return ""
    
    # Convert to uppercase
    text = text.upper()
    
    # Remove all non-alphanumeric characters
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    # Check if plate is too short or too long
    if len(text) < 3 or len(text) > 10:
        return ""
        
    return text 