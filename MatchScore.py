# import nltk
# import re
# from nltk.corpus import stopwords
# from ImageClassifier import InceptionImageClassifier
# import itertools
# import logging
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class MatchScore:
#     def __init__(self, cooccurrence_freq_file, image_tag_freq_file, title_token_freq_file):
#         self.cooccurrence_freq = self.file_to_dictionary(cooccurrence_freq_file)
#         self.image_tag_freq = self.file_to_dictionary(image_tag_freq_file)
#         self.title_token_freq = self.file_to_dictionary(title_token_freq_file)
#         self.classifier = InceptionImageClassifier()
#         self.stopwords = set(stopwords.words('english'))
#         self.stopwords.add("'s")
#
#     def file_to_dictionary(self, file_path):
#         count_dict = {}
#
#         try:
#             with open(file_path, 'r') as f:
#                 for line in f:
#                     term_count = line.split("|")
#                     count_dict["".join(term_count[:-1]).rstrip()] = int(term_count[-1])
#             f.close()
#         except Exception:
#             logger.error("Failed to read File: ", exc_info=True)
#
#         return count_dict
#
#     def inception_classify(self, image_url):
#         return self.classifier.run_inference_on_image(image_url)
#
#     def pmi(self, image_token_pair):
#
#         tag_freq = self.image_tag_freq.get(image_token_pair[1], 0)
#         token_freq = self.title_token_freq.get(image_token_pair[0], 0)
#         cooccurrence_freq = self.cooccurrence_freq.get(str(image_token_pair), 0)
#
#         if cooccurrence_freq > 1:
#             return float(cooccurrence_freq) / (tag_freq * token_freq)
#         else:
#             return False
#
#     def get_score(self, image_url, offer_title):
#
#         try:
#             offer_tokens = [x.encode('utf-8') for x in list(set(nltk.word_tokenize(offer_title.lower()))) if
#                             re.search('[a-zA-Z]', x.encode('utf-8')) and x.encode('utf-8') not in self.stopwords]
#
#         except Exception:
#             logger.error("Failed to parse offer title: ", exc_info=True)
#
#         raw_tags = self.inception_classify(image_url)
#
#         if raw_tags:
#
#             # remove confidence score
#             tags = [x[0] for x in raw_tags]
#
#             # get all (image tag, offer title token) pairs
#             all_pairs = list(itertools.product(offer_tokens, tags))
#
#             # get pmi for all pairs
#             pmis = []
#             for pair in all_pairs:
#                 pmi = self.pmi(pair)
#                 if pmi:
#                     pmis += [pmi]
#
#             if pmis:
#                 return sum(pmis) / len(pmis)
#             else:
#                 return "No co-occurrences found"
#
#         else:
#             print "could not classify image"
#
