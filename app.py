from flask import Flask
from flask import request

from image_feature import ImageEmbedder
from parser import QueryParser

app = Flask(__name__)

@app.route("/")
def home():
    ret = "<img "
    ret += "src = \"https://i.ytimg.com/vi/BzwTRQ1rCUc/maxresdefault.jpg\">"
    return ret

@app.route("/invoke")
def invoke():
    query = request.args.get('query', default='', type=str)
    imgpath = request.args.get('imgpath', default='', type=str)
    print("Processing Query: " + query + " Imgpath: " + imgpath)
    cand_list = app.embedder.getCandidateImages(imgpath, k=50)
    attr_list = app.parser.parseQuery(query)

    re_cand_list = list()
    for cand in cand_list:
        id, dis = cand
        totscore = float(dis)
        if attr_list is not None and len(attr_list) > 0:
            for curattr in attr_list:
                img_attr_score = app.parser.attr_table[id][int(curattr['attr_id'])]
                totscore -= 5*img_attr_score
        re_cand_list.append((id, totscore))

    re_cand_list.sort(key=lambda tup: tup[1])

    ret = ""
    for x in re_cand_list:
        ret += app.parser.image_path[x[0]]
        ret += ','

    return ret


if __name__ == "__main__":
    app.embedder = ImageEmbedder(model_path='/Users/hahaschool/Downloads/model_2.pth')
    app.parser = QueryParser()
    app.parser.load_attributes()
    app.run(debug=True)


