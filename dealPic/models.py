from django.db import models

# Create your models here.

from django.forms import forms

class IMG(models.Model):
    img = models.ImageField(upload_to='img')
    name = models.CharField(max_length=20)

