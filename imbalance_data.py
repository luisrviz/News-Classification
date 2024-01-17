import pandas as pd
from deep_translator import GoogleTranslator
from imblearn.under_sampling import RandomUnderSampler


class Sampling:

    @staticmethod
    def translation(text):
        german_translation = GoogleTranslator(source='auto', target='de').translate(text)
        english_translation = GoogleTranslator(source='de', target='en').translate(german_translation)
        return english_translation

    def oversampling(self, df_train_x, df_train_y):
        df_train_x.reset_index(drop=True, inplace=True)
        df_train_y.reset_index(drop=True, inplace=True)
        df_train = pd.concat([df_train_x, df_train_y], axis=1)
        # In the data exploration we found out that classes A, B, C, I and H have little observations. Let's fix that
        # We will duplicate the observations of those 4 classes.
        training_set_a = df_train.loc[df_train['category'] == "A"]
        training_set_b = df_train.loc[df_train['category'] == "B"]
        training_set_c = df_train.loc[df_train['category'] == "C"]
        training_set_i = df_train.loc[df_train['category'] == "I"]
        training_set_h = df_train.loc[df_train['category'] == "H"]
        for obs in training_set_a["text"]:
            df_train = pd.concat(
                [df_train, pd.DataFrame.from_records([{"text": self.translation(obs), "category": "A"}])])
        for obs in training_set_b["text"]:
            df_train = pd.concat(
                [df_train, pd.DataFrame.from_records([{"text": self.translation(obs), "category": "B"}])])
        for obs in training_set_c["text"]:
            df_train = pd.concat(
                [df_train, pd.DataFrame.from_records([{"text": self.translation(obs), "category": "C"}])])
        for obs in training_set_i["text"]:
            df_train = pd.concat(
                [df_train, pd.DataFrame.from_records([{"text": self.translation(obs), "category": "I"}])])
        for obs in training_set_h["text"]:
            df_train = pd.concat(
                [df_train, pd.DataFrame.from_records([{"text": self.translation(obs), "category": "H"}])])
        print(df_train)
        return df_train[["text"]], df_train[["category"]]


    @staticmethod
    def undersampling(df_train_x, df_train_y):
        # This is done based on the performed exploratory data analysis. We reduce by half the majority class G.
        strategy = {'A': 2000, 'B': 1999, 'C': 2000, 'D': 9999, 'E': 5000, 'F': 9999, 'G': 10000, 'H': 1999,
                    'I': 2000, 'J': 10000}
        rus = RandomUnderSampler(random_state=0, replacement=True, sampling_strategy=strategy)
        x_resampled, y_resampled = rus.fit_resample(df_train_x, df_train_y)
        return x_resampled, y_resampled
