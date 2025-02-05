import cv2
import imageio

def display_sign_gif():
    gif_path = "gifs/hello.gif"  # Adjust path to where your hello.gif is stored
    gif = imageio.mimread(gif_path)  # Load GIF frames
    
    for i, frame in enumerate(gif):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Test GIF Display", img)
        if cv2.waitKey(200) & 0xFF == ord('q'):  # Wait and exit on 'q'
            break

    cv2.destroyAllWindows()

display_sign_gif()
