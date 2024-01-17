from read_data import read_data
from preprocess_data_lr import TFIDF, Vectorizer, CleanTextLr
from preprocess_data_nn import Tokenization, CleanTextNn
from save_results import save_plot_history, save_classification_report, save_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, Bidirectional, SimpleRNN, MaxPooling1D, Conv1D, Dropout, LSTM
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.losses import SparseCategoricalCrossentropy


if __name__ == '__main__':
    # Introduce the path, relative or absolute, to the folder containing the data
    df_train_x, df_train_y, df_test_x, df_test_y, label_encoding = read_data(data_path="")

    # We define the different pipelines for each of the three models
    
    # Logistic regression with TF-IDF
    pipeline_lr = Pipeline([("clean_text_lr", CleanTextLr()), ("vectorizer", Vectorizer()), ("TFIDF", TFIDF()),
                            ("train", LogisticRegression(class_weight="balanced"))])
    pipeline_lr.fit(df_train_x, df_train_y)
    df_test_y_pred = pipeline_lr.predict(df_test_x)
    # We save the classification report
    save_classification_report(df_test_y, df_test_y_pred, "lr")
    # We save the confusion matrix
    save_confusion_matrix(df_test_y, df_test_y_pred, "lr")

    # Neural Network
    def create_model(meta, vocab_size=40340):
        model2 = Sequential()
        model2.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=meta["X_shape_"][1]))
        model2.add(Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.10, activation='tanh', return_sequences=True)))
        model2.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.20, activation='tanh', return_sequences=True)))
        model2.add(Bidirectional(SimpleRNN(64, dropout=0.2, recurrent_dropout=0.20, activation='tanh', return_sequences=True)))
        model2.add(Conv1D(72, 3, activation='relu'))
        model2.add(MaxPooling1D(2))
        model2.add(SimpleRNN(64, activation='tanh', dropout=0.2, recurrent_dropout=0.20, return_sequences=True))
        model2.add(GRU(64, recurrent_dropout=0.20, recurrent_regularizer='l1_l2'))
        model2.add(Dropout(0.2))
        model2.add(Dense(10, activation='softmax'))
        model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
        return model2

    # We wrap the model to make it admisible to the pipeline class_weight="balanced"
    neural_classifier = KerasClassifier(build_fn=create_model, epochs=10, batch_size=128, validation_split=0.2,
                                        verbose=True, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', mode='min',
                                                                               patience=3, min_delta=0.01)])
    pipeline_nn = Pipeline([("clean_text_nn", CleanTextNn()), ("Tokenizer", Tokenization()),
                            ("train", neural_classifier)])
    df_train_y_nn = to_categorical(df_train_y.category, 10)
    pipeline_nn.fit(df_train_x, df_train_y_nn)
    df_test_y_pred = pipeline_nn.predict(df_test_x)
    df_test_y_pred=np.argmax(df_test_y_pred,axis=1)
    # We save the classification report
    save_classification_report(df_test_y, df_test_y_pred, "nn")
    # We save the confusion matrix
    save_confusion_matrix(df_test_y, df_test_y_pred, "nn")
    # We save the learning process
    save_plot_history(pipeline_nn[-1].history_)
    
    # BERT model
    model_name = 'distilbert-base-uncased'
    tam_batch = 16
    n_epochs = 1
    tokenizador = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizador(list(df_train_x["text"].values), truncation=True, padding=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), list(df_train_y["category"].values)))
    model = TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,
                                                                 num_labels=10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                  loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(train_dataset.shuffle(len(train_dataset)).batch(tam_batch), epochs=n_epochs, batch_size=tam_batch)

    test_encodings = tokenizador(list(df_test_x["text"].values), truncation=True, padding=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), list(df_test_y["category"].values)))
    preds = model.predict(test_dataset)["logits"]
    class_preds = np.argmax(preds, axis=1)
    save_classification_report(df_test_y["category"], class_preds, "bert")
