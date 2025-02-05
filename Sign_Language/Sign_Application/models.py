from django.db import models

# Create your models here.

class Registration(models.Model):
    username=models.CharField(unique=True,max_length=100,null=True,blank=True)
    Email=models.EmailField(unique=True,max_length=100,null=True,blank=True)
    Password=models.CharField(max_length=8,null=True,blank=True)
    Confirm_Password=models.CharField(max_length=8,null=True,blank=True)

    def __str__(self):
        return self.username