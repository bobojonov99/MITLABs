# Generated by Django 2.2.7 on 2019-11-30 18:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('orders', '0008_auto_20191130_2059'),
    ]

    operations = [
        migrations.CreateModel(
            name='Payment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('payment_name', models.CharField(blank=True, default=None, max_length=64, null=True)),
                ('data_registration', models.DateField(blank=True, default=None, null=True)),
                ('nds', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('price_nds', models.DecimalField(decimal_places=2, default=0, max_digits=10)),
                ('comments', models.TextField(blank=True, default=None, null=True)),
                ('created', models.DateField(auto_now_add=True)),
                ('updated', models.DateField(auto_now=True)),
                ('order_name', models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.CASCADE, to='orders.Order')),
            ],
            options={
                'verbose_name': 'Платежка',
                'verbose_name_plural': 'Платежки',
            },
        ),
    ]