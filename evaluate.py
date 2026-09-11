import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# Load data
# -----------------------------
train = pd.read_csv("amazonhelp_weak_training.csv")
golden = pd.read_csv("amazonhelp_golden_200.csv")

X_train = train["clean_text"].fillna("")
y_train = train["weak_intent"]

X_test = golden["clean_text"].fillna("")
y_test = golden["intent"]


# -----------------------------
# Baseline 1: Majority class
# -----------------------------
majority_label = y_test.value_counts().idxmax()
majority_pred = [majority_label] * len(y_test)

majority_acc = accuracy_score(y_test, majority_pred)
majority_f1 = f1_score(
    y_test,
    majority_pred,
    average="macro"
)


# -----------------------------
# Baseline 2: Keyword rules
# -----------------------------
def assign_intent_v2(text):
    text = str(text).lower()

    if (
        ("delivered" in text or "marked delivered" in text)
        and any(x in text for x in [
            "not received", "didn't receive",
            "did not receive", "haven't received",
            "can't find", "cannot find",
            "not here", "wrong door",
            "wrong address"
        ])
    ):
        return "delivered_not_received"

    if "prime" in text and any(x in text for x in [
        "membership", "member", "charge",
        "charged", "fee", "subscription",
        "renew", "renewal"
    ]):
        return "prime_membership"

    if any(x in text for x in [
        "account hacked", "hacked", "unauthorized",
        "password", "login", "can't log in",
        "cannot log in", "payment failed",
        "credit card", "debit card",
        "two step verification"
    ]):
        return "account_payment_security"

    if any(x in text for x in [
        "cancel order", "cancel my order",
        "want to cancel", "cancel the order",
        "order cancelled", "order canceled"
    ]):
        return "order_cancellation"

    if any(x in text for x in [
        "refund", "refunded", "money back",
        "return", "returned", "returning"
    ]):
        return "return_refund"

    if any(x in text for x in [
        "damaged", "broken", "wrong item",
        "wrong product", "missing item",
        "defective", "arrived damaged",
        "wrong color", "wrong size",
        "wrong model", "replacement"
    ]):
        return "damaged_wrong_item"

    if any(x in text for x in [
        "fire stick", "fire tv", "echo",
        "alexa", "prime video", "app",
        "not working", "doesn't work",
        "doesnt work", "error message",
        "website", "can't access",
        "cannot access", "won't open",
        "play", "installation"
    ]):
        return "technical_product_support"

    if any(x in text for x in [
        "seller", "marketplace",
        "seller not responding",
        "contact the seller",
        "third party seller"
    ]):
        return "seller_marketplace"

    if any(x in text for x in [
        "in stock", "out of stock",
        "stock", "available",
        "availability", "specifications",
        "specs", "feature", "pre-order",
        "ebook", "kindle", "product",
        "offer", "cashback"
    ]):
        return "product_information"

    if any(x in text for x in [
        "late", "delayed", "overdue",
        "hasn't arrived", "has not arrived",
        "still waiting", "missing package",
        "lost package", "supposed to arrive",
        "due yesterday", "due today",
        "didn't arrive", "not delivered",
        "delivery failed"
    ]):
        return "late_missing_delivery"

    if any(x in text for x in [
        "when will", "delivery date",
        "delivery time", "arrive",
        "shipping date", "where is my order",
        "track my order", "tracking",
        "out for delivery", "in transit",
        "shipped", "dispatch"
    ]):
        return "delivery_status"

    if any(x in text for x in [
        "terrible", "worst", "unhappy",
        "unhelpful", "complaint",
        "poor service", "customer service",
        "not satisfied", "pathetic",
        "useless"
    ]):
        return "complaint_general"

    return "other"


rule_pred = golden["clean_text"].apply(assign_intent_v2)

rule_acc = accuracy_score(y_test, rule_pred)
rule_f1 = f1_score(
    y_test,
    rule_pred,
    average="macro"
)


# -----------------------------
# Main model: TF-IDF + Logistic Regression
# -----------------------------
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

model_pred = model.predict(X_test)

model_acc = accuracy_score(y_test, model_pred)
model_f1 = f1_score(
    y_test,
    model_pred,
    average="macro"
)


# -----------------------------
# Print results
# -----------------------------
print("\n==============================")
print("AMAZONHELP EVALUATION")
print("==============================")

print("\nMajority baseline")
print(f"Accuracy : {majority_acc:.4f}")
print(f"Macro F1 : {majority_f1:.4f}")

print("\nRule baseline")
print(f"Accuracy : {rule_acc:.4f}")
print(f"Macro F1 : {rule_f1:.4f}")

print("\nTF-IDF + Logistic Regression")
print(f"Accuracy : {model_acc:.4f}")
print(f"Macro F1 : {model_f1:.4f}")