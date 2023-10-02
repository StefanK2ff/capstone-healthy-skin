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

    # Get dimensions
    width, height = image.size

    # Determine the the shorter dimension
    shorter_dimension = min(width, height)

    # Compute the left and upper coordinates for cropping
    left = (width - shorter_dimension) / 2
    top = (height - shorter_dimension) / 2
    right = (width + shorter_dimension) / 2
    bottom = (height + shorter_dimension) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    
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


def center_crop_image(np_image) -> np.ndarray:
    """_summary_
    Center crop an image to a square and resize it to the target size

    Args:
        np_image (np.ndarray): image to be cropped
    Returns:
        np.ndarray : cropped image
    """

    # Convert numpy array to PIL Image, rescaling assumes image being rescaled beforehand
    image = Image.fromarray((np_image * 255).astype(np.uint8))
    
    # Calculate dimensions
    width, height = image.size

    # Determine the size of the side to be kept (the shorter side)
    new_dimension = min(width, height)

    # Compute the left and upper coordinates for cropping
    left = (width - new_dimension) / 2
    top = (height - new_dimension) / 2
    right = (width + new_dimension) / 2
    bottom = (height + new_dimension) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    
    # Convert back to numpy array and rescale back
    np_image = np.array(image) / 255.0 # must be float

    return np_image

def resize_as_preprocess(np_image, image_size) -> np.ndarray:
    """_summary_
    Resize an image to the target size

    Args:
        np_image (np.ndarray): image to be resized
        image_size (tuple): target size w*h
    Returns:
        np.ndarray : resized image
    """

    # Convert numpy array to PIL Image, rescaling assumes image being rescaled beforehand
    image = Image.fromarray((np_image * 255).astype(np.uint8))
    
    # Resize the image to the target size
    image = image.resize(image_size)
    
    # Convert back to numpy array and rescale back
    np_image = np.array(image) / 255.0 # must be float

    return np_image