from PIL import Image
import matplotlib.pyplot as plt


# Function to display images based on user input
def show_images(user_input):
    # Define valid image formats
    valid_formats = ['png', 'jpg', 'jpeg']

    # Create a subplot for each image
    fig, axs = plt.subplots(1, len(user_input), figsize=(len(user_input) * 4, 4))

    for i, char in enumerate(user_input):
        if char.isalnum():
            # Iterate through valid image formats
            for img_format in valid_formats:
                img_path = f"D:/Internship Luminar/Main Projects/Sign Language (American)/Sign_Language/ASL/{char.lower()}.{img_format}"

                try:
                    img = Image.open(img_path)
                    axs[i].imshow(img)
                    axs[i].axis('off')  # Turn off axis for cleaner display
                    break  # Break loop if image is found
                except FileNotFoundError:
                    pass  # Continue to the next format if file not found
        elif char.isspace():
            # Display space image
            space_img_path = "Data1/space.jpg"
            img = Image.open(space_img_path)
            axs[i].imshow(img)
            axs[i].axis('off')  # Turn off axis for cleaner display

    plt.show()



# Taking user input
user_input = input("Enter a string: ")
show_images(user_input)
