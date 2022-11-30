# Generated by Django 2.2.7 on 2019-11-28 15:38

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Supplier',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('supplier_name', models.CharField(blank=True, default=None, max_length=64, null=True)),
                ('supplier_country', models.CharField(blank=True, default=None, max_length=64, null=True)),
                ('supplier_address', models.CharField(blank=True, default=None, max_length=128, null=True)),
                ('supplier_phone', models.CharField(blank=True, default=None, max_length=48, null=True)),
                ('supplier_email', models.EmailField(blank=True, default=None, max_length=254, null=True)),
                ('comments', models.TextField(blank=True, default=None, null=True)),
                ('created', models.DateField(auto_now_add=True)),
                ('updated', models.DateField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Производитель',
                'verbose_name_plural': 'Производители',
            },
        ),
    ]
