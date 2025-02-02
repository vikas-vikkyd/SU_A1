import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pickle

label_file_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
test_size = 0.2


def main():
    """Main module to train"""
    # read train data
    df_train_data = pd.read_csv("train_data.csv")

    # read label file
    df_label = pd.read_csv(label_file_path)
    df_label = df_label[["slice_file_name", "classID"]]
    df_label["file_path"] = df_label["slice_file_name"].apply(
        lambda x: x.replace(".wav", ".png")
    )
    df_label = df_label[["file_path", "classID"]]

    # merge feature and label
    df_train_data = df_train_data.merge(df_label, on="file_path", how="inner")
    df_train_data.pop("file_path")

    # train test split
    df_train, df_test = train_test_split(df_train_data, test_size=test_size)

    # y for both train and test
    y_train = df_train.pop("classID")
    y_test = df_test.pop("classID")

    # define cls model
    # cls_model = SVC(gamma="auto", random_state=1234)
    cls_model = RandomForestClassifier(
        n_estimators=100, max_depth=18, min_samples_split=20, random_state=1234
    )
    cls_model.fit(df_train, y_train)

    # check accuracy of model
    print("###################### Model Evaluation ######################\n")
    score = accuracy_score(y_train, cls_model.predict(df_train))
    print("Train accuracy: {}".format(score))
    score = accuracy_score(y_test, cls_model.predict(df_test))
    print("Test accuracy: {}".format(score))
    score = f1_score(y_train, cls_model.predict(df_train), average="weighted")
    print("Train f1 score: {}".format(score))
    score = f1_score(y_test, cls_model.predict(df_test), average="weighted")
    print("Test f1 score: {}".format(score))

    # save model
    filename = 'classify_urban_sound_model.sav'
    pickle.dump(cls_model, open(filename, 'wb'))


if __name__ == "__main__":
    main()
