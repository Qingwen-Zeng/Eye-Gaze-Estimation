from PIL import Image, ImageEnhance
import os
import random

# Path to your original images folder and new folder for augmented images
original_folder = 'train_face'
augmented_folder = 'train_face_au'
# List all files in the original folder
file_list = os.listdir(original_folder)
# Function to create a new directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
# Create the augmented images folder if it doesn't exist
create_directory(augmented_folder)
for filename in file_list:
    img = Image.open(os.path.join(original_folder, filename))
    # Random rotation
    angle = random.randint(-15, 15)
    img = img.rotate(angle)
    # Random scaling and cropping
    scale = random.uniform(0.8, 1.2)
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    img = img.crop((left, top, left + new_width, top + new_height))
    #Random horizontal flipping
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Random changes in brightness and contrast
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.5, 1.5))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.5, 1.5))
    #Random color distortion
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    # Save the augmented image to the new folder
    new_filename = os.path.splitext(filename)[0] + '_augmented.jpg'
    img.save(os.path.join(augmented_folder, new_filename))


print("Augmentation completed!")
