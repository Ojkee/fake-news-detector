import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

nltk.download("english")


class DataFrames:
    def __init__(
        self,
        train_file_name: str,
        test_file_name: str,
        evaluation_file_name: str,
    ) -> None:
        self.content_col_name: str = "content"
        self.content_clean_col_name: str = "content_clean"
        self.content_clean_col_name_no_UNK: str = "content_clean_no_UNK"
        self.content_clean_col_name_truc: str = "content_clean_truc"
        self.label_col_name: str = "label"
        self.train: pd.DataFrame = self.load_data(train_file_name)
        self.test: pd.DataFrame = self.load_data(test_file_name)
        self.evaluation: pd.DataFrame = self.load_data(evaluation_file_name)

    def load_data(self, file_name: str) -> pd.DataFrame:
        df = pd.read_csv(f"dataset/{file_name}.csv", sep=";")
        df.dropna(inplace=True)
        if not self.content_clean_col_name in df.columns:
            df[self.content_col_name] = df["title"] + " " + df["text"]
            df = df.drop(columns=["Unnamed: 0"])
        return df

    def get_datasets(self) -> list[pd.DataFrame]:
        return [self.train, self.test, self.evaluation]

    def get_info(self) -> str:
        test_info = self.test.shape
        train_info = self.train.shape
        evaluation_info = self.evaluation.shape
        return f"DataFrame Shapes:\n\tTrain: {train_info}\n\tTest: {test_info}\n\tEvaluation: {evaluation_info}\n"

    def save_clean(self, token_limit: int = 10000, num_words_trunc: int = 256) -> None:
        # CLEAN
        self._init_clean_content([self.train, self.test, self.evaluation])
        self.train = self.clean_df(self.train)
        self.test = self.clean_df(self.test)
        self.evaluation = self.clean_df(self.evaluation)

        most_common_words = self.get_most_common_words_counter(
            [self.test, self.train],
            token_limit,
            self.content_clean_col_name,
        )
        self.train = DataFrames.set_least_common_UNK(
            self.train, "content_clean", most_common_words
        )
        self.test = DataFrames.set_least_common_UNK(
            self.test, "content_clean", most_common_words
        )
        self.evaluation = DataFrames.set_least_common_UNK(
            self.evaluation, "content_clean", most_common_words
        )
        self.train = DataFrames.drop_least_common(
            self.train,
            self.content_clean_col_name,
            self.content_clean_col_name_no_UNK,
            most_common_words,
        )
        self.test = DataFrames.drop_least_common(
            self.test,
            self.content_clean_col_name,
            self.content_clean_col_name_no_UNK,
            most_common_words,
        )
        self.evaluation = DataFrames.drop_least_common(
            self.evaluation,
            self.content_clean_col_name,
            self.content_clean_col_name_no_UNK,
            most_common_words,
        )
        self.train = DataFrames.trunc_text(
            self.train,
            self.content_clean_col_name_no_UNK,
            self.content_clean_col_name_truc,
            num_words_trunc,
        )
        self.test = DataFrames.trunc_text(
            self.test,
            self.content_clean_col_name_no_UNK,
            self.content_clean_col_name_truc,
            num_words_trunc,
        )
        self.evaluation = DataFrames.trunc_text(
            self.evaluation,
            self.content_clean_col_name_no_UNK,
            self.content_clean_col_name_truc,
            num_words_trunc,
        )
        print(self.test.columns)
        # SAVE
        cols_to_save_clean: list[str] = [
            self.content_clean_col_name,
            self.content_clean_col_name_no_UNK,
            self.content_clean_col_name_truc,
            self.label_col_name,
        ]
        self.train.to_csv(
            f"dataset/train_clean.csv",
            sep=";",
            columns=cols_to_save_clean,
        )
        self.test.to_csv(
            f"dataset/test_clean.csv",
            sep=";",
            columns=cols_to_save_clean,
        )
        self.evaluation.to_csv(
            f"dataset/evaluation_clean.csv",
            sep=";",
            columns=cols_to_save_clean,
        )

    def num_unique_words(self, col_name: str) -> int:
        result: set = set()
        df = pd.concat([self.train, self.test, self.evaluation], ignore_index=True)
        df[col_name].str.lower().str.split().apply(result.update)
        return len(result)

    def get_vocab(self, col_name) -> dict[str, int]:
        vectorizer = CountVectorizer()
        for df in [self.train, self.test, self.evaluation]:
            vectorizer.fit_transform(df[col_name].values)
        return vectorizer.vocabulary_

    def _init_clean_content(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        for df in dfs:
            df[self.content_clean_col_name] = df[self.content_col_name]
        return dfs

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.to_lower(df, self.content_clean_col_name)
        df = self.remove_punctuation(df, self.content_clean_col_name)
        df = self.remove_stopword(df, self.content_clean_col_name)
        return df

    @staticmethod
    def to_lower(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        df[col_name] = df[col_name].apply(lambda x: str(x).lower())
        return df

    @staticmethod
    def remove_punctuation(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        re_punctuation = f'[{re.escape(string.punctuation)}"”“]'
        df[col_name] = df[col_name].apply(
            lambda x: re.sub(re_punctuation, " ", str(x))
            .lower()
            .replace("'s", "")
            .replace("’s", "")
        )
        return df

    @staticmethod
    def remove_stopword(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        stop_words = set(stopwords.words("english"))
        df[col_name] = df[col_name].apply(
            lambda x: " ".join(
                word for word in str(x).split() if not word in stop_words
            )
        )
        return df

    @staticmethod
    def get_most_common_words_counter(
        dfs: list[pd.DataFrame],
        token_limit: int,
        col_name: str,
    ) -> Counter:
        word_counter: Counter = Counter()
        for df in dfs:
            if col_name not in df.columns:
                raise ValueError("Each DataFrame must have a 'clean_content' column")
            tokens = " ".join(df[col_name].astype(str)).split()
            word_counter.update(tokens)
        return Counter(dict(word_counter.most_common(token_limit)))

    @staticmethod
    def set_least_common_UNK(
        df: pd.DataFrame,
        col_name: str,
        most_common_words: Counter,
    ) -> pd.DataFrame:
        df[col_name] = df[col_name].apply(
            lambda x: " ".join(
                [
                    word if word in most_common_words else "<UNK>"
                    for word in str(x).split()
                ]
            )
        )
        return df

    @staticmethod
    def drop_least_common(
        df: pd.DataFrame,
        col_name: str,
        col_name_no_unk: str,
        most_common_words: Counter,
    ) -> pd.DataFrame:
        df[col_name_no_unk] = df[col_name].apply(
            lambda x: " ".join(
                [word for word in str(x).split() if word in most_common_words]
            )
        )
        return df

    @staticmethod
    def trunc_text(
        df: pd.DataFrame,
        col_name: str,
        col_name_trunc: str,
        trunc_num: int,
    ) -> pd.DataFrame:
        df[col_name_trunc] = df[col_name].apply(
            lambda x: " ".join(str(x).split()[:trunc_num])
        )
        return df

    @staticmethod
    def label_to_str(label: int) -> str:
        return "Fake" if label == 1 else "Not Fake"
