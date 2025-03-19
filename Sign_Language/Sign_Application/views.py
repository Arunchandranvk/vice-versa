from django.shortcuts import render,redirect
from django.http import HttpResponse,JsonResponse
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import matplotlib.pyplot as plt
from django.contrib import messages
import time
from PIL import Image
from Sign_Application.models import Registration
from django.views.generic import TemplateView
from PIL import Image
import os
from django.core.mail import send_mail
import random
import string
from django.utils import timezone
from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse
from .forms import SpeechForm
import speech_recognition as sr
from PIL import Image
import matplotlib.pyplot as plt
from django.contrib import messages
import matplotlib
from django.views.decorators import gzip
from collections import Counter
from .animation import *

matplotlib.use('Agg')

def Home(req):
    return render(req,"home.html")

# class VideoCamera:
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         self.detector = HandDetector(maxHands=1)
#         self.classifier = Classifier(r"my_model.h5", r"labels.txt")
#         self.offset = 20
#         self.imgSize = 451
#         self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
#         self.predictions = []
#         self.word=[]
#         self.start_time = time.time()
#         self.letter_interval = 10  # Detect a letter every 2 seconds

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         success, img = self.video.read()
#         if not success:
#             return None

#         imgOutput = img.copy()
#         hands, img = self.detector.findHands(img)

#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand['bbox']
#             imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
#             imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

#             if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
#                 aspectRatio = h / w
#                 if aspectRatio > 1:
#                     k = self.imgSize / h
#                     wCal = math.ceil(k * w)
#                     imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
#                     wGap = math.ceil((self.imgSize - wCal) / 2)
#                     imgWhite[:, wGap:wCal + wGap] = imgResize
#                 else:
#                     k = self.imgSize / w
#                     hCal = math.ceil(k * h)
#                     imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
#                     hGap = math.ceil((self.imgSize - hCal) / 2)
#                     imgWhite[hGap:hCal + hGap, :] = imgResize

#                 prediction, index = self.classifier.getPrediction(imgWhite)
                
#                 # Display the predicted letter on the frame
#                 cv2.putText(imgOutput, f"{self.labels[index]}", 
#                             (x, y - 20), 
#                             cv2.FONT_HERSHEY_COMPLEX, 
#                             2, 
#                             (255, 0, 255), 
#                             2)

#                 # Capture a prediction every 2 seconds
#                 current_time = time.time()
#                 if current_time - self.start_time >= self.letter_interval:
#                     self.predictions.append(self.labels[index])
#                     self.start_time = current_time  # Reset the time to capture the next letter
#                     print(f"Detected Letter: {self.labels[index]} | Word so far: {''.join(self.predictions)}")

#         # Show the current word on the frame
#         current_word = ''.join(self.predictions)
#         self.word.append(current_word)
#         # print("current word",current_word)
#         # self.predictions = current_word
#         cv2.putText(imgOutput, f"Word: {current_word}", 
#                     (50, 50), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     1.5, 
#                     (0, 255, 0), 
#                     3)

#         ret, jpeg = cv2.imencode('.jpg', imgOutput)
#         return jpeg.tobytes()

#     def get_word(self):
#         """Return the detected word."""
#         try:
#             if len(self.predictions) > 0:
#                 word = ''.join(self.word)
#                 return word
#         except Exception as e:
#             print(f"Error in get_word: {e}")
#         return ''



# def hand_signal_detection(request):
#     camera = VideoCamera()
#     start_time = time.time()
#     detection_time = 15  # Detect for 15 seconds

#     while time.time() - start_time < detection_time:  
#         camera.get_frame()

#     word = camera.get_word()
#     # del camera

#     print(f"Final Word: {word}")
#     return render(request, 'video.html', {'predictions': word})

# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         if frame is None:
#             break
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @gzip.gzip_page
# def video_feed(request):
#     try:
#         return StreamingHttpResponse(gen(VideoCamera()), 
#                                      content_type='multipart/x-mixed-replace; boundary=frame')
#     except Exception as e:
#         print(f"Streaming error: {e}")
#         return HttpResponse("Streaming error")
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from django.http import StreamingHttpResponse
from django.shortcuts import render
import time

# ✅ Load the trained model
model_path = "Sign_Language/my_model.h5"
try:
    model = load_model(model_path, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

# ✅ Define class labels
class_labels = {i: chr(65 + i) for i in range(26)}  # A-Z mapping

# ✅ Initialize hand detector
detector = HandDetector(maxHands=1)  # Detect one hand at a time
imgSize = 400  # Match training image size

# ✅ Sentence formation variables
sentence = ""
current_word = ""
last_predicted_letter = None
last_detection_time = time.time()
cooldown_period = 2.5
confirmation_threshold = 7
confirmation_counter = 0

def generate_frames():
    global sentence, current_word, last_predicted_letter, last_detection_time, confirmation_counter
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        hands, _ = detector.findHands(frame, draw=False)  # Detect hands (no drawing)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # White background
        predicted_letter = ""

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            bbox = hand["bbox"]

            if len(lmList) == 0:
                continue

            x_min, y_min = np.min(lmList, axis=0)[:2]
            x_max, y_max = np.max(lmList, axis=0)[:2]

            w, h = x_max - x_min, y_max - y_min
            scale = (imgSize * 0.8) / max(w, h)
            lmList = np.array(lmList)[:, :2]
            lmList = (lmList - [x_min, y_min]) * scale
            lmList += (imgSize - np.array([w, h]) * scale) / 2

            for x, y in lmList.astype(int):
                cv2.circle(imgWhite, (x, y), 8, (0, 0, 0), -1)

            connections = [[0, 1], [1, 2], [2, 3], [3, 4],
                           [0, 5], [5, 6], [6, 7], [7, 8],
                           [0, 9], [9, 10], [10, 11], [11, 12],
                           [0, 13], [13, 14], [14, 15], [15, 16],
                           [0, 17], [17, 18], [18, 19], [19, 20]]

            for p1, p2 in connections:
                cv2.line(imgWhite, tuple(lmList[p1].astype(int)), tuple(lmList[p2].astype(int)), (0, 0, 0), 4)

            imgWhite_gray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            img_input = cv2.resize(imgWhite_gray, (128, 128)) / 255.0
            img_input = np.expand_dims(img_input, axis=-1)
            img_input = np.expand_dims(img_input, axis=0)

            try:
                prediction = model.predict(img_input)
                predicted_label = np.argmax(prediction)
                predicted_letter = class_labels.get(predicted_label, "")
                cv2.putText(frame, f"{predicted_letter}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                if predicted_letter == last_predicted_letter:
                    confirmation_counter += 1
                else:
                    confirmation_counter = 0

                if confirmation_counter >= confirmation_threshold:
                    current_time = time.time()
                    if current_time - last_detection_time > cooldown_period:
                        current_word += predicted_letter
                        last_detection_time = current_time
                        print(f"🔍 Confirmed Letter: {predicted_letter}")
                    confirmation_counter = 0

                last_predicted_letter = predicted_letter

            except Exception as e:
                print(f"❌ Error during prediction: {e}")
                break

        height, width, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.putText(frame, f"Sentence: {sentence}", (50, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Word: {current_word}", (50, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# def index(request):
#     return render(request, 'asl_detection.html')

def RegistrationForm(req):
    return render(req,"Registration.html")

# def get_detected_word(request):
#     """API endpoint to fetch the detected word."""
#     camera = VideoCamera()
#     word = camera.get_word()
#     return JsonResponse({'word': word})

def Registration_save(request):
    if request.method == "POST":
        nm = request.POST.get('uname')
        em = request.POST.get('email')
        passw = request.POST.get('password')
        con = request.POST.get('cpassword')
        if passw != con:
            messages.error(request, "Password and confirm password do not match.")
            return redirect(RegistrationForm)
        registration = Registration(username=nm, Email=em, Password=passw, Confirm_Password=con)
        registration.save()
        messages.success(request,"Registered Succesfully")
        return redirect(Login_Pg)

def Login_Pg(req):
    return render(req,"Login_Pg.html")



def Login_fun(request):
    if request.method=="POST":
        nm=request.POST.get('email')
        pwd=request.POST.get('password')
        if Registration.objects.filter(Email=nm,Password=pwd).exists():
            request.session['Email']=nm
            request.session['Password']=pwd
            messages.success(request, "Logged in Successfully")
            return redirect(Home)
        else:
            messages.warning(request, "Check Your Credentials")
            return redirect(Login_Pg)
    else:
        messages.warning(request, "Check Your Credentials Or Sign Up ")
        return redirect(Login_Pg)

def Logout_fn(request):
    del request.session['Email']
    del request.session['Password']
    messages.success(request, "Logged Out Successfully")
    return redirect(Login_Pg)
            
            

import io
import base64


def show_imagess(request):
    if request.method == 'POST':
        user_input = request.POST.get('text', '').lower()
        valid_formats = ['png', 'jpg', 'jpeg']
        images = []
        fig, axs = plt.subplots(1, len(user_input), figsize=(len(user_input) * 4, 4))
        if len(user_input) == 1:
            axs = [axs]
        
        for i, char in enumerate(user_input):
            if char.isalnum():
                for img_format in valid_formats:
                    img_path = f"ASL/{char.lower()}.{img_format}"
                    
                    try:
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        axs[i].imshow(img)
                        axs[i].axis('off')  
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        images.append(image_base64)
                        
                        break  
                    except FileNotFoundError:
                        pass  
                    except Exception as e:
                        print(f"Error processing image for character {char}: {e}")
            elif char.isspace():
                space_img_path = f"ASL/space.jpg"
                try:
                    img = Image.open(space_img_path)
                    img = img.convert('RGB')                   
                    axs[i].imshow(img)
                    axs[i].axis('off')
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    images.append(image_base64)
                except Exception as e:
                    print(f"Error processing space image: {e}")
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  
        return render(request, 'text_to_sign.html', {'images': images, 'input_text': user_input})
    return render(request, 'text_to_sign.html')





def show_images(user_input):
    # Define valid image formats
    valid_formats = ['png', 'jpg', 'jpeg']

    # Create a subplot for each image
    fig, axs = plt.subplots(1, len(user_input), figsize=(len(user_input) * 4, 4))

    for i, char in enumerate(user_input):
        if char.isalnum():
            # Iterate through valid image formats
        
            for img_format in valid_formats:
                img_path = f"/Sign_Language/ASL/{char.lower()}.{img_format}"

                try:
                    img = Image.open(img_path)
                    
                    axs[i].imshow(img)
                    axs[i].axis('off')  # Turn off axis for cleaner display
                    break  # Break loop if image is found
                except FileNotFoundError:
                    pass  # Continue to the next format if file not found
        elif char.isspace():
            # Display space image
            space_img_path = f"/Sign_Language/ASL/space.jpg"
            img = Image.open(space_img_path)
            axs[i].imshow(img)
            axs[i].axis('off') 
    plt.show()




class Learning_page(TemplateView):
    template_name = "learning_page.html"
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        mp=f"ASL"
        files = os.listdir(mp)

        # Filter out image files (assuming jpg, jpeg, and png formats)
        image_files = [f for f in files if f.endswith(('jpg', 'jpeg', 'png'))]

        # Create absolute paths for each image
        image_paths = [os.path.join(mp, f) for f in image_files]

        # Pass image paths to the template context
        context['image_paths'] = image_paths
        return context
    

class Live(TemplateView):
    template_name = "video.html"

class Animation(TemplateView):
    template_name = "animation.html"


class Text_to_SignLanguage(TemplateView):
    template_name='text_to_sign.html'


def forgetpassword_enteremail(request):
    if (request.method == "POST"):
        email = request.POST['email']

        try:
            user=Registration.objects.get(Email=email)
        except Registration.DoesNotExist:
            user=[]

        if user:
            # Generate a random 6-character OTP
            otp = ''.join(random.choices(string.digits, k=6))

            # Store OTP and timestamp in the session
            request.session['otp'] = otp
            request.session['otp_timestamp'] = timezone.now().timestamp()
            request.session['email'] = email

            # Send the OTP to the user's email
            subject = 'Password Reset OTP'
            message = f'Your OTP for password reset is: {otp}.valid upto 5 minutes'
            from_email = 'jipsongeorge753@gmail.com'
            to_email = email
            send_mail(subject, message, from_email, [to_email], fail_silently=False)
            return redirect('otp')
        else:
            messages.error(request, "invalid user email")

    return render(request,'forget_password_enter_email.html')

def otp(request):
    if (request.method == "POST"):
        otp= request.POST['otp']
        stored_otp = request.session.get('otp')
        stored_email = request.session.get('email')
        timestamp = request.session.get('otp_timestamp')
        print(timestamp)

        # Verify the OTP
        if stored_otp == otp and timezone.now().timestamp() - timestamp <= 300:
            # OTP is valid and not expired (within 5 minutes)
            # You can add additional logic here if needed
            return redirect('new_password')
        elif timestamp is not None:
            if timezone.now().timestamp() - timestamp >= 300:
                request.session.pop('otp', None)
                request.session.pop('otp_timestamp', None)
                request.session.pop('email', None)
                messages.error(request, "invalid OTP")

        else:
            # Clear the session data if OTP is invalid or expired

            messages.error(request, "invalid OTP")

    return render(request,'forget_password_enter_otp.html')


def newpassword(request):
    if request.method == "POST":
        new_password= request.POST['new_password']
        stored_otp = request.session.get('otp')
        stored_email = request.session.get('email')
        timestamp = request.session.get('otp_timestamp')
        user = Registration.objects.get(Email=stored_email)
        if user is None:
            messages.error(request, "invalid user")
        user.Password=new_password
        user.Confirm_Password=new_password
        user.save()
        request.session.pop('otp', None)
        request.session.pop('otp_timestamp', None)
        request.session.pop('email', None)
        return redirect('Login_Pg')
    return render(request, 'new_password.html')


def TextToAnimation(request):
    gif_paths = None
    if request.method == "POST":
        text = request.POST.get('text','')
        if text:
            simplified_text = process_text(text)  # Process the input text
            print("Processed Text:", simplified_text)
            words = simplified_text.split()
            gif_paths = [display_sign_gif(word) for word in words if display_sign_gif(word)]
            print(gif_paths)
    return render(request, 'animation.html', {'animation': gif_paths})

    
class MainHomeView(TemplateView):
    template_name = "mainhome.html"