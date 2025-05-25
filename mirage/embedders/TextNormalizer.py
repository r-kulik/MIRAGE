import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer  # Стеммер для русского и английского
from nltk.stem import WordNetLemmatizer  # Лемматизатор для английского
from natasha import MorphVocab  # Лемматизатор для русского
from typing import Literal


class TextNormalizer:
    """Model to define the algorithm of text normalization. Any derived class can use its inherited functionality"""

    def __init__(
        self,
        stop_word_remove: bool = False,
        word_generalization: Literal["stem", "lemmatize"] | None = "stem",
    ):
        self.stop_word_remove = stop_word_remove
        self.word_generalization = word_generalization
        self.russian_stopwords = set(stopwords.words("russian"))
        self.english_stopwords = set(stopwords.words("english"))
        self.russian_stemmer = SnowballStemmer("russian")  # Стеммер для русского
        self.english_stemmer = SnowballStemmer("english")  # Стеммер для английского
        self.russian_lemmatizer = MorphVocab()  # Лемматизатор для русского
        self.english_lemmatizer = WordNetLemmatizer()  # Лемматизатор для английского

    def normalize(self, text: str) -> str:
        """Returns the normalized text.

        Args:
            text (str): Input text.

        Returns:
            str: Normalized text.
        """
        # Приведение текста к нижнему регистру
        text = text.lower()

        # Удаление пунктуации и специальных символов
        text = re.sub(r"[^\w\s]", " ", text)

        # Токенизация текста на слова
        words = word_tokenize(text, language="russian")

        # Удаление стоп-слов (если требуется)
        if self.stop_word_remove:
            words = [
                word
                for word in words
                if word not in self.russian_stopwords
                and word not in self.english_stopwords
            ]

        # Обобщение слов (лемматизация или стемминг)
        if self.word_generalization == "lemmatize":
            words = [self._lemmatize_word(word) for word in words]
        elif self.word_generalization == "stem":
            words = [self._stem_word(word) for word in words]

        # Сбор текста обратно в строку
        normalized_text = " ".join(words)

        return normalized_text

    def _lemmatize_word(self, word: str) -> str:
        """Лемматизация слова в зависимости от языка."""
        if self._is_russian(word):
            return self.russian_lemmatizer.parse(word).lemma
        elif self._is_english(word):
            return self.english_lemmatizer.lemmatize(word)
        return word

    def _stem_word(self, word: str) -> str:
        """Стемминг слова в зависимости от языка."""
        if self._is_russian(word):
            return self.russian_stemmer.stem(word)
        elif self._is_english(word):
            return self.english_stemmer.stem(word)
        return word

    def _is_russian(self, word: str) -> bool:
        """Проверяет, является ли слово русским."""
        return bool(re.search("[а-яА-Я]", word))

    def _is_english(self, word: str) -> bool:
        """Проверяет, является ли слово английским."""
        return bool(re.search("[a-zA-Z]", word))

    def __call__(self, *args, **kwds):
        return self.normalize(*args, **kwds)
