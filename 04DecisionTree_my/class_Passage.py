#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-11 14:59:21
# @Author  : Lowson Pan (pan.692@osu.edu)

import string
import re
import os
from bs4 import BeautifulSoup


from settings import WEIGHT_TITLE, Only_Passage_With_Topic
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# from nltk.stem.snowball import SnowballStemmer


def load_file_paths():
    data_dir = "../../HW1/02 Codes/reuters/"
    # data_dir = os.getcwd() + os.sep + "reuters" + os.sep
    for filename in os.listdir(data_dir):
        if not os.path.isfile(data_dir + filename):
            continue
        yield data_dir + filename

def parse_passage_from_file(file_name):
    """
    @file_name: input file name.
    @return: iterator of string, contains a single passage's content
    """
    with open(file_name, "r") as fh:
        passage_list = []
        passage = ""
        isInPassge = False
        for line in fh:
            if re.match("<REUTERS.*?>.*", line):
                isInPassge = True
            if isInPassge:
                passage += line
            if re.match(".*?</REUTERS>.*?", line):
                yield passage
                isInPassge = False
                del passage
                passage = ""

def create_passage_obj_list():
    """
    Load and parseall files, then create  & put all passage objects into list.
    @return: list of Passage objects
    """
    cnt = 0
    psg_list = []
    for filepath in load_file_paths():
        tmp_cnt = 0
        for passage in parse_passage_from_file(filepath):
            tmp_cnt += 1
            psg = Passage(passage)
            if Only_Passage_With_Topic and len(psg.topics) == 0:
                continue
            if len(psg.term_counter) < 10:
                continue
            psg_list.append(psg)
            cnt += 1
        # break
        print "[Parser] File %s has been parsed, which have %d passages." % (filepath[-9:], tmp_cnt)
    print "[Parser] Total number of passages parsed: %d. "  % cnt
    return psg_list

class Passage(object):
    def __init__(self, content):
        """
        @params:
            conent: str,  raw passage content parsed from file.
        """
        self.data = self._parse(content)
        self.new_id = int(self.data["reuters"]["newid"])

        # self.new_id = self.data["reuters"]["newid"]
        self.topics = map( lambda s: str(s) , self.data["topics"] )
        self.places = self.data["places"]
        self.term_counter = self.term_filter_NLTK()
        del self.data

    def get_term_counter(self, all_words):
        term_counter = {}
        for word in all_words:
            term_counter[word] = term_counter.get(word, 0) + 1
        return term_counter

    def term_filter_NLTK(self):
        """
        Tokenize -> Stemming -> remove stopwords.
        """
        text_body = self._get_text_body()
        # tokenize
        all_words = word_tokenize(text_body, language='english')
        # filter stopwords & stemming
        stop_words = set(stopwords.words("english"))
        stop_words.add(string.punctuation)
        ps = PorterStemmer()
        # ps = SnowballStemmer("english", ignore_stopwords=True)
        term_counter = {}
        for word in all_words:
            word = word.lower()
            word = ps.stem(word)
            if word in stop_words:
                continue
            term_counter[word] = term_counter.get(word, 0) + 1
        return term_counter

    def _filter_punctuation_number(self, text_body):
        # filtering numbers & punctuations.
        filtered_body = ""
        unallowed = set(list(string.punctuation) + list(string.digits))
        # filtering punctuation and number
        for char in text_body:
            if char in unallowed:
                filtered_body += " "
                continue
            filtered_body += char
        return filtered_body

    def _parse(self, content):
        """
        Helper Function.
        Time: O(Number of tags)
        Parse a single passage in xml format into dictionary.
        @return: dictionary of dictionary, which has same structure as the original XML file
        """
        data = {}
        soup = BeautifulSoup(content, "html.parser")
        data["reuters"] = soup.reuters.attrs
        data["places"] = list(soup.places.strings)
        data["topics"] = list(soup.topics.strings)
        for tag in soup.reuters.contents:
            if tag == "\n":
                continue
            data[tag.name] = list(tag.strings)
            if tag.name == "text":
                tmp = {}
                for child_tag in tag.contents:
                    tmp[child_tag.name] = child_tag.string
                data["text"] = tmp
        return data

    def _get_text_body(self):
        text_body = (self.data["text"].get("title", "") + " ") * WEIGHT_TITLE +  self.data["text"].get("body", "")
        return self._filter_punctuation_number(text_body)

    def cal_TF_IDF(self, feature_vector_attrs, IDF):
        attrs_set = set(feature_vector_attrs)
        word_set = self.get_word_set()
        # TF
        total_term_count = 0
        for word, cnt in self.term_counter.items():
            if word in attrs_set:
                total_term_count += cnt
        total_term_count = float(total_term_count)
        # TF-IDF
        # TFs = []
        TF_IDFs = []
        for attr in feature_vector_attrs:
            if attr not in self.term_counter:
                # TFs.append(0)
                TF_IDFs.append(0)
                continue
            TF = self.term_counter.get(attr, 0) / total_term_count
            # TFs.append(round(TF, 3))
            TF_IDFs.append( round(TF * IDF[attr], 4) )
        return TF_IDFs


    def get_word_set(self):
        return set( self.term_counter.keys() )

    def __str__(self):
        s = ""
        for key, val in self.data["TEXT"].items():
            s += repr(key) +":         :       :" + repr(val) + "\n"
        return s

# Testing code
def test():
    print "[Passage] Testing Code."
    cnt = 0
    for filepath in load_file_paths():
        tmp_cnt = 0
        for passage in parse_passage_from_file(filepath):
            obj = Passage(passage)
            cnt += 1
            tmp_cnt += 1
            print obj.new_id
            if cnt == 3:
                break
        print "[Parser] %s has been parsed, which have %d files." % (filepath[-10:], tmp_cnt)
        break
    print "[Parser] Total number of passages parsed: %d. "  % cnt
    print "[Info] Number of passges Parsed:", cnt

if __name__ == "__main__":
    test()
