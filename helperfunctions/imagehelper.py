# helperfunctions/imagehelper.py


from PIL import Image
import numpy as np
import string 
import random

def img_load_and_transform(image_path, target_size) -> np.ndarray:

    """Load an image from disk and resize it to the target size and returns it as a NumPy array.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size of the image.
    Returns:
        np.ndarray: Processed image.
        
    """ 
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Resize the image to the target size
    image = image.resize(target_size)
    
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)
    
    return image_array



def generate_random_string(length=6) -> str:

    """_summary_
    Generate a random string of fixed length

    Args:
        length (int, optional): length of the random string. Defaults to 6.
    Returns:
        str : random string of letters and digits
    """
    letters_and_digits = string.ascii_letters + string.digits

    return ''.join(random.choice(letters_and_digits) for i in range(length))