import pickle

# load the model from disk
loaded_model = pickle.load(open('model.pkl', 'rb'))

model = loaded_model.predict([[2,3,1,2,1]])

model = ''.join(model)

print(model)
