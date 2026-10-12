# Spam Classifier

A small SMS spam detector built with scikit-learn. It downloads the public SMS spam dataset, trains a TF-IDF + logistic regression pipeline, evaluates it with cross-validation and a confusion matrix, and then classifies messages you type in, with a confidence score for each.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python spam_classifier.py
```

On the first run the script will:

1. Download the dataset (`sms.tsv`) from the [pycon-2016-tutorial](https://github.com/justmarkham/pycon-2016-tutorial) repo.
2. Run 5-fold stratified cross-validation on the full dataset and print the mean and standard deviation of accuracy, precision, recall and F1.
3. Split the data 80/20 (stratified) and train a `TfidfVectorizer` + `LogisticRegression` pipeline on the 80%.
4. Print a classification report and a confusion matrix for the held-out 20%.
5. Save the trained model to `spam_model.pkl`.

It then lets you check as many messages as you like. Each one prints `SPAM` or `HAM` with the model's confidence, for example `SPAM (94%)`. A lower percentage, such as `SPAM (60%)`, means the message is borderline. Enter a blank line or `quit` to exit.

Later runs load the saved model, so they start instantly and need no internet. To rebuild the model (for example after changing the code):

```bash
python spam_classifier.py --retrain
```

Only load `spam_model.pkl` files you created yourself, since pickle files can run code when loaded.

## Run on Google Colab

Colab already includes pandas, requests and scikit-learn, so there is nothing to install. In a new notebook, run this in a cell:

```python
!git clone https://github.com/dguo97/Simple-Spam-Detector.git
%cd Simple-Spam-Detector
%run spam_classifier.py
```

Alternatively, upload `spam_classifier.py` using the Files panel on the left and run `%run spam_classifier.py`.

Notes:

- Use `%run`, not `!python`. Colab doesn't pass typed input to shell commands, so the message loop would exit immediately. Pasting the script into a cell also fails because it uses `argparse` and `__file__`.
- Type each message into the input box that appears. Press Enter on a blank line to stop.
- To retrain, run `%run spam_classifier.py --retrain`.
- Colab storage is temporary, so `spam_model.pkl` is deleted when the runtime resets and the script retrains on the next run (about a second).

## Example results

5-fold cross-validation on the full dataset (mean +/- std; precision, recall and F1 are for the spam class):

| Metric    | Score           |
|-----------|-----------------|
| Accuracy  | 0.980 +/- 0.004 |
| Precision | 0.930 +/- 0.014 |
| Recall    | 0.921 +/- 0.022 |
| F1        | 0.925 +/- 0.016 |

Hold-out test set (1,115 messages):

|                | Predicted ham | Predicted spam |
|----------------|---------------|----------------|
| **Actual ham** | 953           | 13             |
| **Actual spam**| 13            | 136            |

The classifier uses `class_weight="balanced"` because only about 13% of the messages are spam. Without it, the model was very cautious: it flagged almost no normal messages (precision 0.99) but let about a quarter of spam through (recall 0.73, F1 0.84). Balanced weighting catches far more spam (recall 0.92) at the cost of flagging a few more normal messages as spam (precision 0.93). If wrongly flagging a real message is worse for your use case than missing spam, you can remove `class_weight="balanced"` from `make_pipeline()` in `spam_classifier.py`.

## License

[MIT](LICENSE)

## Requirements

- Python 3.9+
- Internet access for the first run and for `--retrain` (the dataset is downloaded when training)
