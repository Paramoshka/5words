from django.db import models


class Word(models.Model):
    id = models.AutoField(primary_key=True)
    word = models.CharField(max_length=100)
    vector = models.CharField(max_length=100)
