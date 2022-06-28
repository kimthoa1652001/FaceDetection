from os import listdir
from torchvision import transforms
from numpy import dot
from PIL import Image
from numpy.linalg import norm
import cv2
import torch
from numpy import asarray
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def cosin_distance(a,b):
	return dot(a,b) / (norm(a)*norm(b))

def fixed_image_standardization(image_tensor):
	processed_tensor = (image_tensor - 127.5) / 128.0
	return processed_tensor


def trans(img):
	transform = transforms.Compose([
		transforms.ToTensor()])
	return transform(img)
def Brute_Force(img1,img2):
	sift = cv2.SIFT_create()

	kp1,des1 = sift.detectAndCompute(img1,None)
	kp2,des2 = sift.detectAndCompute(img2,None)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	res = 0
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			res+=1
	return res



def extract_face_from_image(image, required_size=(160, 160)):
	# load image and detect faces
	faces = detector.detect_faces(image)
	box = []
	# extract the bounding box from the requested face
	if faces != None:
		x1, y1, width, height = faces[0]['box']
		x2, y2 = x1 + width, y1 + height
		box.extend([x1,y1,x2,y2])
		# extract the face
		face_boundary = image[y1:y2, x1:x2]

		# resize pixels to the model size
		face_image = Image.fromarray(face_boundary)
		face_image = face_image.resize(required_size)
		face_array = asarray(face_image)
	else:
		return None,None
	return box,face_array
def extract_multiface_from_image(image, required_size=(160, 160)):
	# load image and detect faces
	faces = detector.detect_faces(image)
	boxes= []
	faces_array = []
	# extract the bounding box from the requested face
	if faces != None:
		for face in faces:
			box = []
			print(face)
			x1, y1, width, height = face['box']
			x2, y2 = x1 + width, y1 + height
			box.extend([x1,y1,x2,y2])
			# extract the face
			face_boundary = image[y1:y2, x1:x2]

			# resize pixels to the model size
			face_image = Image.fromarray(face_boundary)
			face_image = face_image.resize(required_size)
			face_array = asarray(face_image)

			faces_array.append(face_array)
			boxes.append(box)
	return boxes, faces_array
def extract_features(img):
	embedding = facenet(trans(img).unsqueeze(0))
	#print(embedding)
	return embedding.detach().numpy()
def load_embeding(directory):
	X_data = []
	labels = []
	embeddings = []
	for subdir in listdir(directory):
		paths = directory + subdir + '/'
		detect_face = []
		if subdir != '.DS_Store':
			for filename in listdir(paths):
				if filename == '.DS_Store':
					continue
				path = paths + filename    #name
				print(path)
				img = cv2.imread(path,cv2.COLOR_BGR2RGB)
				#cv2.imshow('im',img)
				boxes, face_image = extract_face_from_image(img)
				detect_face.append(face_image)
			label = [subdir for _ in range(len(detect_face))]
			X_data.extend(detect_face)
			labels.extend(label)
	X_data,labels= asarray(X_data), asarray(labels)
	#print(X_data.shape,labels.shape)
	for face_list in X_data:
		embed = extract_features(face_list)
		embeddings.append(embed)
	return embeddings,labels
def inference(face,train_face,power,threshold=0.5):
	embed = extract_features(face)
	embed = embed.flatten()
	#print(embed.shape)
	#print(embed)
	min_score = abs(cosin_distance(embed,train_face[0].flatten()))
	id = 0
	for idx,emb in enumerate(train_face):
		emb = emb.flatten()
		if min_score<abs(cosin_distance(embed,emb)):
			min_score = abs(cosin_distance(embed,emb))
			id = idx
	print(min_score)
	if min_score<threshold:
		return -1,-1
	else:
		return id,min_score
	"""
	maxx=0
	id = -1
	for idx, emb in enumerate(train_face):
		print(emb.shape)
		score = Brute_Force(embed,emb)
		res = max(score,maxx)
		id = idx
	if res<threshold:
		return -1,-1
	else:
		return id,res
	"""
#Test_Phase
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval()
detector = MTCNN()
embeddings, names = load_embeding('data/')
cap = cv2.VideoCapture(0)
while True:
	sucess,frame = cap.read()
	if sucess:
		bbox,faces = extract_multiface_from_image(frame)
		if bbox is not None:
			for id,face in enumerate(faces):
				idx,score = inference(face,embeddings,power=pow(10,6))
				if idx != -1:
					frame = cv2.rectangle(frame, (bbox[id][0],bbox[id][1]), (bbox[id][2],bbox[id][3]), (0,0,255), 6)
					frame = cv2.putText(frame,names[idx],(bbox[id][0],bbox[id][1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
				else:
					frame = cv2.rectangle(frame,(bbox[id][0],bbox[id][1]), (bbox[id][2],bbox[id][3]), (0,0,255), 6)
					frame = cv2.putText(frame,'Unknown',(bbox[id][0],bbox[id][1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
	cv2.imshow('Face Recognition',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()