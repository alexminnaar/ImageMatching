import os
from sklearn.externals import joblib


class TitleClassifier:
    def __init__(self):
        self.model_path = os.path.dirname(os.path.abspath(__file__)) + "/model/offer_title_classifier_2.pkl"
        self.model = joblib.load(self.model_path)
        self.category_mapping = {1: "arts_and_entertainment", 2: "automotive", 3: "cameras_and_photo", 4: "computing",
                                 5: "consumer_electronics", 6: "family_and_baby", 7: "fashion_and_accessories",
                                 8: "gaming", 9: "home_and_garden", 10: "jewelry_and_watches",
                                 11: "news_books_and_magazines", 12: "pets", 13: "sports_and_fitness",
                                 14: "toys_and_hobbies", 15: "health_and_beauty", 16: "food_and_drink",
                                 17: "cell_phones_and_mobile"}

    def title_classify(self, offer_title):
        return self.category_mapping[self.model.predict(offer_title)[0]]


# def main():
#     cl = TitleClassifier()
#     print cl.title_classify(["ASRock LGA2011-v3/ Intel X99/ DDR4/SATA3&USB3.1/ "])
#
#
# if __name__ == "__main__":
#     main()
