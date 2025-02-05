from django.urls import path
from Sign_Application import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns=[
    path('hand_signal_detection/',views.hand_signal_detection,name="hand_signal_detection"),
    path('home',views.Home,name="Home"),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('show_images/',views.show_imagess,name="show_images"),
    path('Registration_save/',views.Registration_save,name="Registration_save"),

    path('',views.Login_Pg,name="Login_Pg"),
    path('RegistrationForm/',views.RegistrationForm,name="RegistrationForm"),

    path('Login_fun/',views.Login_fun,name="Login_fun"),
    path('Logout_fn/',views.Logout_fn,name="Logout_fn"),
    # path('shows/',views.shows,name='s'),
    path('learn/',views.Learning_page.as_view(),name="learn"),
    path('video/',views.Live.as_view(),name='video'),
    path('text_to_sign/',views.Text_to_SignLanguage.as_view(),name="text_to_sign"),
    path('forget-password/', views.forgetpassword_enteremail, name="forget_password_enter_email"),
    path('otp',views.otp,name='otp'),
    path('new_password',views.newpassword,name='new_password'),
    path('get_detected_word/', views.get_detected_word, name='get_detected_word'),
    path('animation/', views.Animation.as_view(), name='animation'),
    path('text-animation/', views.TextToAnimation, name='text-animation'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
