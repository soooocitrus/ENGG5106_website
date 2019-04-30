import sys
import os
import ast
import json
import numpy as np
import operator
from itertools import islice
from scipy import spatial

def generateInputImageInfo(inputfilename):
	###The tmp.txt contains image info
	###The format is 'image_path|potential_categaries|image_vector'
	###Please kindly notice that:
	###when we run get_input_image_feature.py, some addtional modules is needed. 
	###So please change the absulote directory properly, otherwise there would be something wrong.
	os.system('python3 /Users/Citrus/Desktop/bird-classification-recvis/get_input_image_feature.py --data /Users/Citrus/Desktop/ENGG5106/'+inputfilename+' --outfile /Users/Citrus/Desktop/ENGG5106/tmp.txt')

def readInputImageInfo():
	info = dict()
	with open("tmp.txt") as f:
		for line in f:
			items = line.split('|')
			info['image_path'] = items[0]
			info['potential_categories'] = ast.literal_eval(items[1]) 
			info['image_vector'] = ast.literal_eval(items[2])
	return info

def readStoredImageInfo(potential_categories, input_image_vector):
	###read potential categories iamges infos one by one.
	###All infos are stored in infos
	###infos = {image_path : distance}
	base = np.array(input_image_vector)
	infos = dict()
	for category_id in potential_categories:
		with open("features/"+str(category_id)+".txt") as f:
			firstLine = True
			for line in f:
				if firstLine:
					firstLine = False
					continue
				items = line.split('|')
				infos[items[1]] = np.linalg.norm(np.array(ast.literal_eval(items[3])) - base)
	return infos

def getCanidateImages(inputfilename, k):
	###Here is the canidate images after passing though the image filter.
	###inputfilename is the stored input file path, k is top k images would be return.
	###The result is in format: list(zip(image_path, distance))
	generateInputImageInfo(inputfilename)
	inputInfo = readInputImageInfo()
	storedInfos = readStoredImageInfo(inputInfo['potential_categories'], inputInfo['image_vector'])
	sorted_storedInfos = sorted(storedInfos.items(), key=operator.itemgetter(1))
	#print(sorted_storedInfos)
	k_items = list(islice(sorted_storedInfos,k))
	return k_items

def getPicName(inputtext, inputfilename):
	###get pictures here, store the pictures' name in dataToSendBack as list
	#print(inputfilename)
	dataToSendBack = []
	# for i in range(10):
	# 	dataToSendBack.append(str(i) + ".jpg")
	###	

	###Here is the canidate images after passing though the image filter.
	###The result is in format: list(zip(image_path, distance))
	candidateImages = getCanidateImages(inputfilename, 10)
	for image_path, distance in candidateImages:
		dataToSendBack.append(image_path)
	return dataToSendBack

def main(inputtext,inputfilename):
	return getPicName(inputtext, inputfilename);

if __name__ == '__main__':
	dataToSendBack = main(sys.argv[1], sys.argv[2])
	print(sys.argv[2])
	print("answeris")
	for i in range(10):
		print(dataToSendBack[i])		
		print(" ")
	sys.stdout.flush()


