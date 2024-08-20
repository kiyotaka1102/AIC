from pyvi import ViUtils, ViTokenizer
import googletrans
import translate
from difflib import SequenceMatcher
from langdetect import detect
#from underthesea import sent_tokenize, text_normalize, classify, sentiment


class Text_Preprocessing():
    def __init__(self, stopwords_path=f"./data/vietnamese-stopwords-dash.txt"):
        with open(stopwords_path, 'r', encoding='utf-8') as f:  # Open in text mode for easier string handling
            self.stop_words = [line.strip() for line in f.readlines()]

    def find_substring(self, string1, string2):
        match = SequenceMatcher(None, string1, string2, autojunk=False).find_longest_match(0, len(string1), 0, len(string2))
        return string1[match.a:match.a + match.size].strip()

    def remove_stopwords(self, text):
        text = ViTokenizer.tokenize(text)
        filtered_words = [w for w in text.split() if w not in self.stop_words]
        return " ".join(filtered_words)

    def lowercasing(self, text):
        return text.lower()

    def uppercasing(self, text):
        return text.upper()

    def add_accents(self, text):
        return ViUtils.add_accents(text)

    def remove_accents(self, text):
        return ViUtils.remove_accents(text)

    # def sentence_segment(self, text):
    #     return sent_tokenize(text)

    # def text_norm(self, text):
    #     return text_normalize(text)

    # def text_classify(self, text):
    #     return classify(text)

    # def sentiment_analysis(self, text):
    #     return sentiment(text)

    def __call__(self, text):
        # Apply preprocessing steps
        text = self.lowercasing(text)
        #text = self.remove_stopwords(text)
        # Uncomment and adjust as needed
        # text = self.remove_accents(text)
        # text = self.add_accents(text)
        # text = self.text_norm(text)
        return text  # Return the processed text

class Translation():
    def __init__(self, from_lang='vi', to_lang='en', mode='google'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate.
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang
        self.text_processing = Text_Preprocessing()
        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang,to_lang=to_lang)

    def preprocessing(self, text):

        return self.text_processing(text) #text.lower()

    def __call__(self, text):

        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
                else self.translator.translate(text, dest=self.__to_lang).text

