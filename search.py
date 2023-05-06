from flask import Flask, request, render_template, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

app.debug = True
class SearchEngine:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer()
        self.index = self.vectorizer.fit_transform(documents)

    def search(self, query):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.index)[0]
        results = [(i, score) for i, score in enumerate(scores)]
        results = sorted(results, key=lambda x: -x[1])
        return results

documents = [
    """มาริโอ้" โชว์หวาน "จันจิ" คือ คน เดียว ใน ใจ รัก 8 ปี จับมือ แฟน ก้าวผ่าน คำ บูลลี่""",
    "ไอซ์ ปรีชญา เคลียร์ ปม สั่ง ไซยาไนด์ กำจัด สัตว์ มี พิษ ร่ำไห้ ถูกโยงอดีต ผู้ จัดการ เสียชีวิต",
    "“มาริโอ้” ไม่ ซีเรียส แต่ เสียใจ ถูก ตั้งกระทู้ วิจารณ์ รูปร่าง",
    "มาริโอ้ วิ่งไปหา จันจิ ถ่ายรูป ด้วยกัน สุด มุ้งมิ้ง เสิร์ฟ โมเมนต์ หวาน หาดูยาก",
    "ประวัติ ไอซ์ ปรีชญา พงษ์ธนานิกร"
]

engine = SearchEngine(documents)

@app.route('/')
def index():
    return render_template('search.html')

@app.route('/search')
def search():
    query = request.args.get('query')
    print('query = ',query)
    results = engine.search(query)
    print('result = ',results)
    return render_template('results.html', query=query, results=results, documents=documents)

if __name__ == '__main__':
    app.run(debug=True)
