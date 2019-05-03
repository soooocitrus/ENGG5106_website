import argparse
import PIL.Image as Image
from Model.models import resnet101
import torch
import torch.nn as nn
import numpy as np
from Model.tools import data_transformer
import ast
import operator
from itertools import islice

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='input.png', metavar='D',
                    help="input image file to be processed.")
parser.add_argument('--model', type=str, default='model.pth', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='input_feature.txt', metavar='D',
                    help="output directory of the images features")



class ImageEmbedder(object):
    def __init__(self, model_path = './model.pth'):
        self.use_cuda = torch.cuda.is_available()

        print("Loading model: " + model_path)
        self.state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model, self.input_size = resnet101()
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

        self.model_feature = nn.Sequential(*list(self.model.children())[:-1])
        for param in self.model_feature.parameters():
            param.requires_grad = False

        if self.use_cuda:
            print('Using GPU')
            self.model.cuda()
        else:
            print('Using CPU')

        self.data_transforms = data_transformer(self.input_size)


    def pil_loader(self, path):
            # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            with open(path, 'rb') as f:
                with Image.open(f) as img:
                    return img.convert('RGB')

    def get_embedding(self, datafile, outfile):
        output_file = open(outfile, "w+")
        # output_file.write("Path|Categories|Feature\n")
        data = self.data_transforms(self.pil_loader(datafile))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if self.use_cuda:
            data = data.cuda()
        output = np.reshape(self.model(data).data, 200)
        feature = np.reshape(self.model_feature(data),(2048))
        pred = np.argsort(-output)[:5]
        output_file.write("%s|%s|%s\n" % (datafile, str(pred.tolist()), str(feature.tolist())))
        output_file.close()

        print("Succesfully wrote " + outfile + ', you can find input image info in it.')

    def generateInputImageInfo(self, inputfilename):
        self.get_embedding(inputfilename, 'tmp.txt')

    def readInputImageInfo(self):
        info = dict()
        with open("tmp.txt") as f:
            for line in f:
                items = line.split('|')
                info['image_path'] = items[0]
                info['potential_categories'] = ast.literal_eval(items[1])
                info['image_vector'] = ast.literal_eval(items[2])
        return info

    def readStoredImageInfo(self, potential_categories, input_image_vector):
        ###read potential categories iamges infos one by one.
        ###All infos are stored in infos
        ###infos = {image_path : distance}
        base = np.array(input_image_vector)
        infos = dict()
        for category_id in potential_categories:
            if category_id == 0:
               continue
            with open("Model/features/" + str(category_id) + ".txt") as f:
                firstLine = True
                for line in f:
                    if firstLine:
                        firstLine = False
                        continue
                    items = line.split('|')
                    infos[items[0]] = 1-np.dot(np.array(ast.literal_eval(items[3])),base)/(np.linalg.norm(np.array(ast.literal_eval(items[3])))*np.linalg.norm(base))
        return infos

    def getCandidateImages(self, inputfilename, k):
        ###Here is the canidate images after passing though the image filter.
        ###inputfilename is the stored input file path, k is top k images would be return.
        ###The result is in format: list(zip(image_path, distance))
        self.generateInputImageInfo(inputfilename)
        inputInfo = self.readInputImageInfo()
        storedInfos = self.readStoredImageInfo(inputInfo['potential_categories'], inputInfo['image_vector'])
        sorted_storedInfos = sorted(storedInfos.items(), key=operator.itemgetter(1))
        # print(sorted_storedInfos)
        k_items = list(islice(sorted_storedInfos, k))
        return k_items

    def getPicName(self, inputtext, inputfilename):
        ###get pictures here, store the pictures' name in dataToSendBack as list
        # print(inputfilename)
        dataToSendBack = []

        ###Here is the canidate images after passing though the image filter.
        ###The result is in format: list(zip(image_path, distance))
        candidateImages = self.getCandidateImages(inputfilename, 10)
        for image_path, distance in candidateImages:
            dataToSendBack.append(image_path)
        return dataToSendBack

if __name__ == "__main__":
    print("Testing Image Enbedding Loader...")
    embedder = ImageEmbedder(model_path='/Users/hahaschool/Downloads/model_2.pth')

    res = embedder.getCandidateImages(inputfilename='Model/bird_dataset/train_images/007.Parakeet_Auklet/Parakeet_Auklet_0004_795921.jpg',
                                k=10)
    print(res)


