from django.shortcuts import redirect, render  
from .form import ImageForm  


def image_request(request):  
    if request.method == 'POST':  
        form = ImageForm(request.POST, request.FILES)  
        if form.is_valid():  
            form.save()
            # print(form)
            # Getting the current instance object to display in the template  
            img_object = form.instance

            return render(request, 'image_form.html', {'form': form, 'img_obj': img_object})  
    else:  
        form = ImageForm()  
        return render(request, 'image_form.html', {'form': form})  