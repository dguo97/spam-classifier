# Spam Classifier

A small SMS spam detector built with scikit-learn. It trains a logistic regression model on word and character TF-IDF features, then classifies messages you type in, with a confidence score for each.

---

## 🚀 How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click **+ New notebook**.
3. Run the following in a cell:

```
# Clone repository and enter project directory
!git clone https://github.com/dguo97/spam-classifier.git
%cd spam-classifier

# Run script (type messages into the input box that appears)
%run spam_classifier.py
```

Use `%run`, not `!python`: Colab doesn't pass typed input to shell commands, so the message loop would exit immediately. Colab storage is temporary, so the model is retrained (about 10 seconds) after the runtime resets.

## 💻 Run Locally

Requires Python 3.9+ and internet access on the first run.

```bash
pip install -r requirements.txt
python spam_classifier.py
```

Each message prints `SPAM` or `HAM` with a confidence, for example `SPAM (94%)`. A lower percentage, such as `SPAM (60%)`, means the message is borderline. Enter a blank line or `quit` to exit.

The trained model is saved to `spam_model.pkl` and reused on later runs. To rebuild it, run `python spam_classifier.py --retrain`. Only load `.pkl` files you created yourself, since pickle files can run code when loaded.

---

## 📊 Dataset Overview

The project uses the public **SMS Spam Collection**, downloaded automatically as `sms.tsv` from the [pycon-2016-tutorial](https://github.com/justmarkham/pycon-2016-tutorial) repo.

- **Classes:** `ham` (normal) and `spam`.
- **Volume:** 5,572 messages, about 13% of them spam.

---

## 🧠 How It Works

1. **Features:** TF-IDF on single words and word pairs, combined with character n-grams, which pick up spammy spellings such as "FREE!!!" or "claim4".
2. **Model:** `LogisticRegression` with balanced class weights, so the rare spam class isn't ignored.
3. **Evaluation:** 5-fold stratified cross-validation, plus a classification report and confusion matrix on a stratified 80/20 hold-out split.
4. **Cutoff:** a message is labelled spam at 50% probability or higher. Change this in `classify()`.

---

## 📈 Results

5-fold cross-validation (mean ± std; precision, recall and F1 are for the spam class):

| Metric    | Score           |
|-----------|-----------------|
| Accuracy  | 0.990 ± 0.003   |
| Precision | 0.975 ± 0.012   |
| Recall    | 0.950 ± 0.014   |
| F1        | 0.963 ± 0.009   |

Hold-out test set (1,115 messages):

|                 | Predicted ham | Predicted spam |
|-----------------|---------------|----------------|
| **Actual ham**  | 964           | 2              |
| **Actual spam** | 7             | 142            |

---

## 📄 License

[MIT](LICENSE)
