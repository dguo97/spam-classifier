# Spam Classifier

A small SMS spam detector built with scikit-learn. It downloads the public SMS spam dataset, trains a logistic regression model on word and character TF-IDF features, evaluates it with cross-validation and a confusion matrix, and then classifies messages you type in, with a confidence score for each.

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
3. Split the data 80/20 (stratified) and train the model on the 80%. It combines word features (single words and pairs) with character n-grams, followed by a `LogisticRegression` classifier.
4. Print a classification report and a confusion matrix for the held-out 20%.
5. Save the trained model to `spam_model.pkl`.

It then lets you check as many messages as you like. Each one prints `SPAM` or `HAM` with the model's confidence, for example `SPAM (94%)`. A lower percentage, such as `SPAM (60%)`, means the message is borderline. Enter a blank line or `quit` to exit.

Training takes about 10 seconds. Later runs load the saved model, so they start in a couple of seconds and need no internet. To rebuild the model (for example after changing the code):

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
- Colab storage is temporary, so `spam_model.pkl` is deleted when the runtime resets and the script retrains on the next run (about 10 seconds).

## Example results

5-fold cross-validation on the full dataset (mean +/- std; precision, recall and F1 are for the spam class):

| Metric    | Score           |
|-----------|-----------------|
| Accuracy  | 0.990 +/- 0.003 |
| Precision | 0.975 +/- 0.012 |
| Recall    | 0.950 +/- 0.014 |
| F1        | 0.963 +/- 0.009 |

Hold-out test set (1,115 messages):

|                | Predicted ham | Predicted spam |
|----------------|---------------|----------------|
| **Actual ham** | 964           | 2              |
| **Actual spam**| 7             | 142            |

### How the model was improved

Each change was compared using the same 5-fold cross-validation (spam precision / recall / F1):

| Version                                              | Precision | Recall | F1    |
|------------------------------------------------------|-----------|--------|-------|
| Word features only                                   | 0.986     | 0.726  | 0.836 |
| + `class_weight="balanced"`                          | 0.930     | 0.921  | 0.925 |
| + word pairs and character n-grams (current model)  | 0.975     | 0.950  | 0.963 |

- **Balanced class weights.** Only about 13% of the messages are spam, so the plain model was very cautious: it almost never flagged a normal message but let about a quarter of spam through.
- **Character n-grams.** These pick up spammy spellings and symbols such as "FREE!!!", "txt" or "claim4", and they improved both precision and recall.

The script labels a message as spam when the spam probability is at least 50%. You can change the cutoff in `classify()` in `spam_classifier.py`. A higher cutoff flags fewer normal messages but misses more spam: in cross-validation, a cutoff of 80% flagged none of the 4,825 normal messages but missed about 100 of the 747 spam messages.

## License

[MIT](LICENSE)

## Requirements

- Python 3.9+
- Internet access for the first run and for `--retrain` (the dataset is downloaded when training)
