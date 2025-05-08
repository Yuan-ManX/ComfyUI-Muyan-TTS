import unittest
import re
from num2words import num2words

class LangSegment:
    _text_cache = {}
    _text_lasts = None
    _text_langs = None

    PARSE_TAG = re.compile(r'(⑥\$\d+[\d]{6,}⑦)')

    @staticmethod
    def getTexts(text: str) -> list:
        if not text or not text.strip():
            LangSegment._text_langs = None
            return []
        
        if LangSegment._text_lasts == text and LangSegment._text_langs:
            return LangSegment._text_langs
        
        LangSegment._text_lasts = text
        LangSegment._text_cache = {}
        
        words = LangSegment._parse_text(text)
        LangSegment._text_langs = words
        return words

    @staticmethod
    def _parse_text(text: str) -> list:
        if not text:
            return []
        
        process_list = [
            ('$4', re.compile(r'\b[a-zA-Z]+(?:[-_][a-zA-Z]+)*\b'), LangSegment._process_english),
            ('$0', re.compile(r'\b\d+\b'), LangSegment._process_number),
        ]
        
        for tag, pattern, process in process_list:
            text = LangSegment._replace_patterns(text, tag, pattern, process)
        
        return LangSegment._process_tags([], text)

    @staticmethod
    def _replace_patterns(text: str, tag: str, pattern, process) -> str:
        matches = pattern.findall(text)
        if len(matches) == 1 and matches[0] == text:
            return text
        
        for i, match in enumerate(matches):
            key = f'⑥{tag}{i:06d}⑦'
            text = pattern.sub(key, text, count=1)
            LangSegment._text_cache[key] = (process, (tag, match))
        return text

    @staticmethod
    def _process_tags(words: list, text: str) -> list:
        segments = LangSegment.PARSE_TAG.split(text)
        for segment in segments:
            if LangSegment.PARSE_TAG.match(segment):
                process, data = LangSegment._text_cache[segment]
                process(words, data)
            else:
                current_word = ""
                for char in segment:
                    if char in ",.!?\";:": 
                        if current_word:
                            LangSegment._addwords(words, current_word)
                            current_word = ""
                        words.append({"text": char}) 
                    else:
                        current_word += char
                if current_word:
                    LangSegment._addwords(words, current_word)
        return words

    @staticmethod
    def _process_english(words, data):
        _, match = data
        LangSegment._addwords(words, match)

    @staticmethod
    def _process_number(words, data):
        _, match = data
        try:
            number = int(match)
            text = num2words(number, lang='en')
            LangSegment._addwords(words, text)
        except ValueError:
            LangSegment._addwords(words, match)

    @staticmethod
    def _addwords(words, text):
        if not text or not text.strip():
            return
        text = LangSegment._insert_english_uppercase(text)
        words.append({"text": text})

    @staticmethod
    def _insert_english_uppercase(word):
        modified_text = re.sub(r'(?<!\b)([A-Z])', r' \1', word)
        modified_text = re.sub(r'[-_]', ' ', modified_text)
        modified_text = re.sub(r'\s+', ' ', modified_text).strip()
        return modified_text
