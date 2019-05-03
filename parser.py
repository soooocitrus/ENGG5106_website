import os,sys
import nltk
import re
import spacy
import numpy as np

class QueryParser(object):

    def __init__(self):
        super(QueryParser, self).__init__()
        self.attr_path = './attributes/'
        self.image_path = dict()
        self.attr = dict()
        self.partattr = dict()
        self.tokenizer = nltk.tokenize.casual.TweetTokenizer()
        self.stemmer = nltk.stem.SnowballStemmer("english", ignore_stopwords=True)
        self.stop_words = nltk.corpus.stopwords.words('english')
        self.part_names = set()
        self.spacy_entity = spacy.load('en_core_web_lg')
        self.attr_table = dict()
        self.all_attr_word = list()
        self.attr_count = 312

    def load_attributes(self):
        print(self.attr_path)

        # Image Path
        file = open(self.attr_path+'images-dirs.txt', "r")
        for x in file.readlines():
            id, path = x.strip().split()
            self.image_path[id] = path
        file.close()

        # Attribute String
        nam = set()
        file = open(self.attr_path+'attributes.txt', "r")
        for x in file.readlines():
            id, attr_str = x.strip().split(' ', 1)
            cat, det = attr_str.strip().split('::', 1)

            # [Dirty Part] Obtain part names
            part = re.match(r'\b.+?_(.+)::', attr_str).group(1)
            part = part[::-1].split('_', 1)
            if len(part) > 1:
                part = part[1][::-1]
            else:
                part = part[0][::-1]
            self.part_names.add(part)

            self.attr[id] = dict()
            self.attr[id]['part'] = part
            self.attr[id]['category'] = cat
            self.attr[id]['key'] = det

            self.all_attr_word.append(det + ' ' + part)

            if part not in self.partattr.keys():
                self.partattr[part] = dict()
            self.partattr[part][det] = id
        file.close()

        # Image Attribute
        print("Loading attribute table...")

        base_score = 1.0
        conf_score = [1, 0.5, 0.75, 1]


        file = open(self.attr_path+'labels.txt', "r")
        for x in file.readlines():
            img_id, attr_id, prs, cert, wker = x.strip().split()
            if img_id not in self.attr_table.keys():
                self.attr_table[img_id] = [0.0 for i in range(0,self.attr_count+1)]
            if prs == '1':
                self.attr_table[img_id][int(attr_id)] += base_score*conf_score[int(cert)-1]
            else:
                self.attr_table[img_id][int(attr_id)] += base_score * (1.0-conf_score[int(cert)-1])
        print("Done.")

    def parseQuery(self, qry):
        # Segmentation
        if qry is None or len(qry.strip()) == 0:
            return None
        qry_segs = list()
        for curqry in qry.strip().split(','):
            curqry = curqry.strip()
            confidence = 1.0
            # Assign Part
            tkn = [word for word in self.tokenizer.tokenize(curqry) if word not in self.stop_words]
            asig_part = None
            fuzzy_part_score = -114514.0
            fuzzy_part = None
            for curwd in tkn:
                # Exact Match
                if curwd in self.part_names:
                    asig_part = curwd
                    #print("Exact part matched : " + curwd)
                    break
                # Fuzzy Match
                else:
                    qw = self.spacy_entity(curwd)
                    for part in self.part_names:
                        pw = self.spacy_entity(part)
                        sim = qw.similarity(pw)
                        if sim > fuzzy_part_score:
                            fuzzy_part_score = sim
                            fuzzy_part = part
            if asig_part is None:
                #print("Fuzzy part matched : " + fuzzy_part + " (" + str(fuzzy_part_score) + ")")
                asig_part = fuzzy_part
                confidence *= fuzzy_part_score

            # Assign attribute
            asig_attr = None
            asig_attr_id = None
            fuzzy_attr_score = -114514810.0
            fuzzy_attr = None
            for curwd in tkn:
                # Exact match
                if curwd in self.partattr[asig_part].keys():
                    asig_attr = curwd
                    #print("Exact attr matched : " + curwd)
                    break
                # Fuzzy match
                else:
                    qw = self.spacy_entity(curwd)
                    for attr in self.partattr[asig_part].keys():
                        pw = self.spacy_entity(attr)
                        sim = qw.similarity(pw)
                        if sim > fuzzy_attr_score:
                            fuzzy_attr_score = sim
                            fuzzy_attr = attr
            if asig_attr is None:
                #print("Fuzzy attr matched : " + fuzzy_attr + " (" + str(fuzzy_attr_score) + ")")
                asig_attr = fuzzy_attr
                confidence *= fuzzy_attr_score

            asig_attr_id = self.partattr[asig_part][asig_attr]
            #print("Matched: [" + asig_attr_id + "] " + asig_part + ", " + asig_attr + " (" + str(confidence) + ")")
            qry_segs.append({
                "part" : asig_part,
                "attr" : asig_attr,
                "attr_id" : asig_attr_id,
                "confidence" : confidence
            })

        return qry_segs


            # # Pure Fuzzy
            # qrysen = ' '.join(tkn)
            # print(qrysen)
            # qw = self.spacy_entity(qrysen)
            # mx = -1.0
            # ans = None
            # for x in self.all_attr_word:
            #     pw = self.spacy_entity(x)
            #     sim = str(qw.similarity(pw))
            #     if float(sim) > mx:
            #         mx = float(sim)
            #         ans = x
            #
            # print("Pure Fuzzy: " + ans + " : " + str(mx))





if __name__ == '__main__':
    print("Testing Parser")
    parser = QueryParser()
    parser.load_attributes()
    parser.parseQuery("eyes are in red, back is pink, have a leg which is blue, head is blue, orange neck")