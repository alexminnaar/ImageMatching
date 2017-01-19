import nltk
import re
from nltk.corpus import stopwords
from ImageClassifier import ImageClassifier
import urllib
import os
import itertools


class MatchScore:
    def __init__(self, cooccurrence_freq_file, image_tag_freq_file, title_token_freq_file):
        self.cooccurrence_freq = self.file_to_dictionary(cooccurrence_freq_file)
        self.image_tag_freq = self.file_to_dictionary(image_tag_freq_file)
        self.title_token_freq = self.file_to_dictionary(title_token_freq_file)
        self.classifier = ImageClassifier()
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.add("'s")

    def file_to_dictionary(self, file_path):
        count_dict = {}

        with open(file_path, 'r') as f:
            for line in f:
                term_count = line.split("|")
                count_dict["".join(term_count[:-1]).rstrip()] = int(term_count[-1])

        f.close()

        return count_dict

    def inception_classify(self, image_url):
        return self.classifier.run_inference_on_image(image_url)

    def pmi(self, image_token_pair):

        tag_freq = self.image_tag_freq.get(image_token_pair[1], 0)
        token_freq = self.title_token_freq.get(image_token_pair[0], 0)
        cooccurrence_freq = self.cooccurrence_freq.get(str(image_token_pair), 0)

        if cooccurrence_freq > 1:
            return float(cooccurrence_freq) / (tag_freq * token_freq)
        else:
            return False

    def get_score(self, image_url, offer_title):

        offer_tokens = [x.encode('utf-8') for x in list(set(nltk.word_tokenize(offer_title.lower()))) if
                        re.search('[a-zA-Z]', x.encode('utf-8')) and x.encode('utf-8') not in self.stopwords]

        raw_tags = self.inception_classify(image_url)
        print raw_tags

        if raw_tags:

            # remove confidence score
            tags = [x[0] for x in raw_tags]

            # get all (image tag, offer title token) pairs
            all_pairs = list(itertools.product(offer_tokens, tags))

            # get pmi for all pairs
            pmis = []
            for pair in all_pairs:
                pmi = self.pmi(pair)
                if pmi:
                    pmis += [pmi]

            if pmis:
                return sum(pmis) / len(pmis)
            else:
                return "No co-occurrences found"

        else:
            print "could not classify image"


def main():

    # jean example
    jeans_image = "https://images.viglink.com/product/250x250/demandware-edgesuite-net/ef37e28fb846752e958361da835bd2301fc3ebbc.jpg?url=http%3A%2F%2Fdemandware.edgesuite.net%2Fsits_pod43%2Fdw%2Fimage%2Fv2%2FAATD_PRD%2Fon%2Fdemandware.static%2F-%2FSites-cog-men-master%2Fdefault%2Fdw114d49f3%2F390548%2F390548-68-2.jpg%3Fsw%3D520%26sh%3D780%26sm%3Dfit"
    jeans_title = "skinny straight leg/spitfire jeans"

    # incorrect example
    incorrect_image = "https://images.viglink.com/product/250x250/www-forever21-com/5f93b7c66e76d1beb508e4df8694bda3595c0000.jpg?url=http%3A%2F%2Fwww.forever21.com%2Fimages%2Fno_image.jpg"
    incorrect_title = "Velvet Halter Bodycon Dress"

    # phone example
    phone_image = "https://images.viglink.com/product/250x250/ecx-images-amazon-com/d33aa854aaa55854d771b61928f0d4b05551e870.jpg?url=http%3A%2F%2Fecx.images-amazon.com%2Fimages%2FI%2F31nKE1Q9MRL.jpg"
    phone_title = "Spigen Neo Hybrid Crystal Galaxy S7 Edge Case with "

    # lip balm example
    lip_balm_image = "https://images.viglink.com/product/250x250/www-sephora-com/095e00e9c738e84e3c84705c321966fe5756cbaf.jpg?url=http%3A%2F%2Fwww.sephora.com%2Fimages%2Fsku%2Fs1733245-main-hero.jpg"
    lip_balm_title = "Too Cool For School Dinoplatz Lip Balm"

    # gown example
    gown_image = "https://images.viglink.com/product/250x250/cache-net-a-porter-com/c9c6e2f63d1822da211e8d8a73c601866e5a9ac8.jpg?url=https%3A%2F%2Fcache.net-a-porter.com%2Fimages%2Fproducts%2F794963%2F794963_cu_pp.jpg"
    gown_title = "Supernova tiered off-the-shoulder embellished tulle gown"

    # # pencil example
    # pencil_image = "https://images.viglink.com/product/250x250/www-techrabbit-com/b1f9e07584d0da362b48670c99f277cb48550639.jpg?url=http%3A%2F%2Fwww.techrabbit.com%2Fmedia%2Fcatalog%2Fproduct%2Fcache%2F1%2Fimage%2F720x660%2F9df78eab33525d08d6e5fb8d27136e95%2Ff%2Fi%2Ffif-pencil-gp_01_2.jpg"
    # pencil_title = "Pencil by FiftyThree Digital Stylus for iPad and iPhone "

    # shoe example
    shoe_image = "https://images.viglink.com/product/250x250/static-slamjamsocialism-com/40666c4c1220528c30ecaaa70773f08c10aa75e2.jpg?url=http%3A%2F%2Fstatic.slamjamsocialism.com%2F99470%2Fyeezy-boost-350-v2-sneakers.jpg"
    shoe_title = "adidas originals Yeezy Boost 350 V2 Sneakers"

    # incorrect example 2
    incorrect_image_2 = "https://images.viglink.com/product/250x250/www-hm-com/03fe7cc12130e695784638492c618d15bf6d0ee2.jpg?url=http%3A%2F%2Fwww.hm.com%2Fjosh%2Fstatic%2Fsite%2Fimg%2Fcommon%2Fbg_main.png%3F1"
    incorrect_title_2 = "Suede Boots"

    ms = MatchScore(cooccurrence_freq_file="/users/alexminnaar/image_counts/cooccurrence_freq.txt",
                    image_tag_freq_file="/users/alexminnaar/image_counts/image_tag_freq.txt",
                    title_token_freq_file="/users/alexminnaar/image_counts/title_token_freq.txt")

    score = ms.get_score(jeans_image, jeans_title)
    print "jeans score: ", score

    score = ms.get_score(incorrect_image, incorrect_title)
    print "incorrect score: ", score

    score = ms.get_score(phone_image, phone_title)
    print "phone score: ", score

    score = ms.get_score(lip_balm_image, lip_balm_title)
    print "lip balm score: ", score

    score = ms.get_score(gown_image, gown_title)
    print "gown score: ", score

    # score = ms.get_score(pencil_image, pencil_title)
    # print "stylus score: ", score

    score = ms.get_score(shoe_image, shoe_title)
    print "shoe score: ", score

    score = ms.get_score(incorrect_image_2, incorrect_title_2)
    print "incorrect 2 score: ", score


if __name__ == "__main__":
    main()
