from keras.models import load_model
from keras import backend as K

# function for calculating r-square or coefficient-of-determination
def r2_score(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true-y_pred)) 
	SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
	return (1 - SS_res/SS_tot)

model = load_model('model.h5', custom_objects={'r2_score': r2_score})
