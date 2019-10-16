from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os.path import isdir
from os import listdir
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder

def extract_face_util(image, required_size=(160, 160)):
    try:
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    except:
        return []

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    return extract_face_util(image, required_size)
    

def extract_face_live(image, required_size=(160, 160)):
    image = Image.fromarray(image)
    return extract_face_util(image, required_size)

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        try:
            # get face
            face = extract_face(path)
            # store
            faces.append(face)
        except:
            continue
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

# get the face embedding for one face
def get_one_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def get_embeddings(model, data):
    newdata = list()
    for face_pixels in data:
        embedding = get_one_embedding(model, face_pixels)
        newdata.append(embedding)
    return asarray(newdata)
    
    
# normalize input vectors
def normalize(data):
    in_encoder = Normalizer(norm='l2')
    data = in_encoder.transform(data)
    return data

# label encode targets
def label_encode(trainy, testy):
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    return trainy, testy, out_encoder