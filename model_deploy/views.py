from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
import pickle

# Create your views here.

def webpage(request):
    return render(request,'deploy.html')

def values(request):
    
    if request.method == 'POST':
        location = float(request.POST.get('myList'))
        adult_m = float(request.POST.get('content3'))
        adult_f = float(request.POST.get('content4'))
        case_r = float(request.POST.get('content5'))
        true_c = float(request.POST.get('content7'))

        # load the model from disk
        loaded_model = pickle.load(open('Model/model.pkl', 'rb'))
        model = loaded_model.predict([[location,adult_m,adult_f,case_r,true_c]])
        model = ''.join(model)

    return render(request,'deploy.html', {"get_prediction":model})
