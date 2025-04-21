from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load stopwords
stopwords_set = set(stopwords.words('english'))

# Fix regex warning by using raw string
emoticon_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')

app = Flask(__name__)

# Use raw strings to fix Windows path errors
model_path = r'C:\Users\Harish\OneDrive\Desktop\Sentiment-Analysis-Mahcine-Learning-NLP-Project-main\clf.pkl'
vectorizer_path = r'C:\Users\Harish\OneDrive\Desktop\Sentiment-Analysis-Mahcine-Learning-NLP-Project-main\tfidf.pkl'

# Load model and vectorizer
with open(model_path, 'rb') as f:
    clf = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    tfidf = pickle.load(f)

# Preprocessing function
def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(text)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Preprocess and transform comment
        preprocessed_comment = preprocessing(comment)
        comment_vector = tfidf.transform([preprocessed_comment])
        sentiment = clf.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment, comment=comment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
