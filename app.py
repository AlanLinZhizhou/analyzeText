from flask import Flask, jsonify, render_template, request

from stanfordcorenlp import StanfordCoreNLP
import utils.wordpiece as wp
from utils.vocab import Vocab
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
path = 'H:/a/nlp/stanford-corenlp-full-2018-10-05'
nlp = StanfordCoreNLP(path, lang='en')
app = Flask(__name__)
# nltk.download('vader_lexicon')  # 下载情感词典

# 路由解析：通过用户访问的路径，匹配相应函数
@app.route('/')
def index():
    return render_template("index.html")


# 返回首页
@app.route('/index')
def home():
    return render_template("index.html")


# 返回电影页面
@app.route('/movie')
def movie():
    sentence=request.args["sentence"]
    label=request.args["label"]
    print(sentence,label)
    nullPage=[]
    if sentence=='null':
        return render_template("movie.html", movies=nullPage)
    l_dict=[]
    lc='1\tit is a beautiful nice day'
    l_dict.append(lc)
    judges = ''
    triples = ''
    op_dict = {}
    vocab = Vocab()
    vocab.load('utils/google_uncased_en_vocab.txt')
    count_wp = 0
    op_skip_list = []
    ent_skip_list = []

    for i in range(0, len(l_dict)):
        # for i in range(1, 3):

        s_label, s_content = label,sentence
        if s_label == '0':
            judges = 'bad'
        if s_label == '1':
            judges = 'good'
        token = nlp.word_tokenize(s_content)
        if i % 500 == 0:
            print(i+1, ' examples,  ', len(l_dict) - i, 'to go')
        dependencyParse = nlp.dependency_parse(s_content)
        pos = nlp.pos_tag(s_content)
        # a1=nlp.pos_tag('apple')[0][1]
        wptoken = wp.WordpieceTokenizer(vocab.i2w)
        for i, begin, end in dependencyParse:
            # print (i, '-'.join([str(begin), token[begin-1]]), '-'.join([str(end),token[end-1]]))
            if i == 'amod' or i == 'nsubj':
                if nlp.pos_tag(token[begin - 1])[0][1] in ['NN', 'NNS', 'NNP', 'NNPS','JJ', 'JJR', 'JJS'] and \
                        nlp.pos_tag(token[end - 1])[0][1] in ['JJ', 'JJR', 'JJS','NN', 'NNS', 'NNP', 'NNPS']:
                    if nlp.pos_tag(token[begin - 1])[0][1] in ['JJ', 'JJR', 'JJS'] and nlp.pos_tag(token[end - 1])[0][1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                        entity1 = token[end - 1]
                        op = token[begin - 1]
                    else:
                        entity1 = token[begin - 1]
                        op = token[end - 1]
                    entity2 = wptoken.tokenize(entity1)[0]
                    entity2 = entity2.strip()
                    if op.lower() in ['true', 'right']:
                        judges = 'good'
                    if op.lower() in ['false', 'wrong']:
                        judges = 'bad'
                    if len(entity2) > 1:
                        count_wp += 1

                    if entity2 + '\t' + op not in op_dict.keys():
                        good_num, bad_num = 0, 0
                        if judges == 'good':
                            good_num = 1
                        else:
                            bad_num = 1
                        op_dict[entity2 + '\t' + op] = judges + '\t' + '1' + '\t' + '0' + '\t' + str(
                            good_num) + '\t' + str(
                            bad_num)

                    else:
                        ac_judges, count_tr, conf, good_num, bad_num = op_dict[entity2 + '\t' + op].split('\t')
                        count_tr = int(count_tr) + 1
                        if judges == 'good':
                            good_num = int(good_num) + 1
                        if judges == 'bad':
                            bad_num = int(bad_num) + 1
                        # 判断是否有冲突
                        if conf == '1' or judges != ac_judges:
                            op_dict[entity2 + '\t' + op] = judges + '\t' + str(count_tr) + '\t' + '1' + '\t' + str(
                                good_num) + '\t' + str(bad_num)
                        else:
                            op_dict[entity2 + '\t' + op] = judges + '\t' + str(count_tr) + '\t' + '0' + '\t' + str(
                                good_num) + '\t' + str(bad_num)
    # 遍历op_dict，放入到list中去
    messages = []
    for key in op_dict.keys():
        message=[]
        k1,k2=key.split('\t')
        message.append(k1)
        message.append(k2)
        v1,v2,v3,v4,v5=op_dict[key].split('\t')
        if v1=='good':
            v1='positive'
        if v1=='bad':
            v1='negative'
        message.append(v1)
        v2=v2+' time(s)'
        if v3=='0':
            v3='no conflicts'
        if v3=='1':
            v3='conflicted knowledge'
        v4=v4+' vote(s)'
        v5=v5+' vote(s)'
        message.append(v2)
        message.append(v3)
        message.append(v4)
        message.append(v5)
        messages.append(message)
    # datalist = []
    # con = sqlite3.connect("moveTop.db")  # 打开数据库
    # cur = con.cursor()  # 获取游标
    # sql = "select * from movieTop250"  # sql查询语句
    # data = cur.execute(sql)  # 获取数据
    # for item in data:  # 将数据保存在列表中
    #     datalist.append(item)
    # cur.close()  # 关闭游标
    # con.close()  # 关闭连接
    return render_template("movie.html", movies=messages)

# 返回评分界面
@app.route('/score')
def score():
    messages=[]
    with open(r'./kgs/movie.spo','r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            i1,i2,i3,i4,i5,i6,i7,i8=line.split('\t')
            if i3 == 'good':
                i3 = 'positive'
            if i3 == 'bad':
                i3 = 'negative'
            i4 = i4 + ' time(s)'
            if i5 == '0':
                i5 = 'no conflicts'
            if i5 == '1':
                i5 = 'conflicted knowledge'
            i6 = i6 + ' vote(s)'
            i7 = i7 + ' vote(s)'
            if i8[0]!='-':
                i8=i8[0:3]
            else:
                i8=i8[0:4]
            message=[]
            message.append(i1)
            message.append(i2)
            message.append(i3)
            message.append(i4)
            message.append(i5)
            message.append(i6)
            message.append(i7)
            message.append(i8)
            messages.append(message)
    return render_template("score.html",movies=messages)


@app.route('/word')
def word():
    return render_template("word.html")


# 返回词云界面
@app.route('/wordsdeal', methods=['POST'])
def words_deal():
    input_text = request.form['input_text']
    score = get_sentiment(input_text)
    pol = ''
    if score>=0:
        pol = pol + 'positive'
    else:
        pol = pol + 'negative'
    final_str = "该句子的情感极性为："+pol+" ,句子情感得分为:"+str(score)
    return jsonify({'processed_text': final_str})

def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    sentiment_polarity = sentiment_score['compound']
    return sentiment_polarity


# 返回团队界面
@app.route('/team')
def team():
    return render_template("team.html")


if __name__ == '__main__':
    app.run(debug=True)