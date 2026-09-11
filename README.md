# AmazonHelp AI Customer Support Agent

An AI customer-support agent built on the Customer Support on Twitter
(TWCS) dataset, focused on the AmazonHelp brand.

The system combines:

1. Intent classification
2. Historical-response retrieval
3. AUTO-HANDLE vs ESCALATE triage
4. A Streamlit demonstration UI

---

## 1. Problem

Given a customer-support message, the agent predicts the customer's
support intent, retrieves a historically similar AmazonHelp response,
and decides whether the request can be handled automatically or should
be escalated to a human.

The goal is not only to classify a customer message, but to build a
small end-to-end support workflow that is grounded in historical
AmazonHelp behavior.

---

## 2. Architecture

![AmazonHelp AI Support Agent Architecture]![alt text](image.png)

The pipeline has four main stages:

1. **Intent classification** using TF-IDF features and Logistic Regression.
2. **Historical-response retrieval** using TF-IDF similarity over
   historical AmazonHelp customer messages.
3. **Risk and confidence checks** to decide whether automation is safe.
4. **AUTO-HANDLE or ESCALATE** with a historically grounded response.

---

## 3. Intent Taxonomy

The system uses 12 customer-support intents:

- `delivery_status`
- `late_missing_delivery`
- `delivered_not_received`
- `damaged_wrong_item`
- `return_refund`
- `order_cancellation`
- `prime_membership`
- `account_payment_security`
- `technical_product_support`
- `product_information`
- `seller_marketplace`
- `complaint_general`

The taxonomy was created through exploratory analysis of AmazonHelp
messages and manual inspection of representative examples.

---

## 4. Data and Evaluation Setup

The project uses the Customer Support on Twitter (TWCS) dataset and
focuses on AmazonHelp interactions.

The full raw dataset is intentionally **not included** in this
repository.

The repository contains:

- `amazonhelp_golden_200.csv` — 200 hand-labelled golden evaluation examples
- `amazonhelp_weak_training.csv` — 3,000 weakly labelled training examples
- `amazonhelp_response_pairs.csv` — 30,000 derived historical AmazonHelp customer/support-response pairs
- `amazonhelp_response_eval_30.csv` — 30-example response evaluation sample
- `amazonhelp_response_eval_30_to_rate.csv` — response-quality rating template

The 200-example golden set was kept completely separate from model
training.

### Training data

The 3,000-example training set contains up to 250 weakly labelled
examples per intent.

The labels were produced using heuristic rules and are therefore
treated as **weak supervision**, not ground truth.

### Golden evaluation data

The 200-example golden set was manually labelled and is used only for
evaluation.

This separation prevents evaluation leakage into the classifier.

---

## 5. Golden Evaluation Set

The golden set contains 200 manually labelled examples.

| Intent | Examples |
|---|---:|
| `complaint_general` | 51 |
| `late_missing_delivery` | 43 |
| `product_information` | 21 |
| `damaged_wrong_item` | 19 |
| `delivery_status` | 14 |
| `account_payment_security` | 13 |
| `technical_product_support` | 12 |
| `order_cancellation` | 9 |
| `delivered_not_received` | 9 |
| `return_refund` | 7 |
| `seller_marketplace` | 2 |

The evaluation set was not balanced artificially because it represents
the manually labelled sample used for final evaluation.

---

## 6. Intent Classification Results

Three approaches were compared on the same 200-example golden set.

| Approach | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 25.5% | 3.69% |
| Keyword/rule baseline | 27.5% | 33.16% |
| **TF-IDF + Logistic Regression** | **41.5%** | **42.75%** |

The learned classifier improves over both baselines.

The classifier is trained only on the 3,000 weakly labelled training
examples.

The hand-labelled golden set is used only for final evaluation and is
not used for model training.

### Reproduce the benchmark

```bash
python evaluate.py
```

Expected output:

```text
==============================
AMAZONHELP EVALUATION
==============================

Majority baseline
Accuracy : 0.2550
Macro F1 : 0.0369

Rule baseline
Accuracy : 0.2750
Macro F1 : 0.3316

TF-IDF + Logistic Regression
Accuracy : 0.4150
Macro F1 : 0.4275
```

---

## 7. Response Generation

The response layer is retrieval-based rather than free-form generation.

For each customer message, the system searches historical AmazonHelp
customer messages and retrieves the associated historical support reply
from the closest match.

This provides a traceable grounding mechanism and reduces the risk of
inventing unsupported support instructions.

However, text similarity does not guarantee that the retrieved response
solves the exact customer problem.

For example, a delivery-related message may retrieve another
delivery-related response that is textually similar but addresses a
different issue.

The system therefore uses the response-match score together with
classifier confidence when making the automation decision.

---

## 8. AUTO-HANDLE vs ESCALATE

The agent uses a conservative escalation policy.

A request is escalated when any of the following is true:

- model confidence is below `0.60`
- response match score is below `0.25`
- predicted intent is `account_payment_security`
- predicted intent is `seller_marketplace`

Otherwise, the request can be AUTO-HANDLED.

### Golden-set triage results

| Decision | Count | Percentage |
|---|---:|---:|
| AUTO-HANDLE | 7 | 3.5% |
| ESCALATE | 193 | 96.5% |

Among AUTO-HANDLE cases, the predicted intent was correct for:

**71.4% of cases**

The policy is intentionally conservative. It prioritizes avoiding
incorrect automation over maximizing automation coverage.

### Why this matters

An incorrect automated customer-support response can be worse than
sending a case to a human.

The escalation layer therefore combines:

- classifier confidence
- response-match quality
- high-risk intent rules

---

## 9. Failure Analysis

A total of **117 of the 200 golden examples were misclassified** by the
TF-IDF model.

The main failure patterns were:

### 1. Prime mention can override the actual delivery problem

Customers frequently mention Prime while actually reporting a delayed
or missing package.

Example pattern:

```text
Prime membership + package is late
```

The classifier may incorrectly predict `prime_membership` instead of
`late_missing_delivery`.

### 2. Delivery status vs late delivery

Messages asking where an order currently is can be confused with reports
that an order has already become late.

Both categories contain similar words such as:

- package
- order
- delivery
- arrive
- tracking

### 3. Generic complaints become specific intents

Vague dissatisfaction can be classified as a more specific category
such as technical support, delivery, or returns.

### 4. Technical support vs product information

Questions about device or app capabilities can look similar to reports
that a feature or application is not working.

### 5. Multi-intent messages

Customers often put multiple requests into one message, such as:

- delivery problem + refund
- delivery problem + cancellation
- wrong item + refund
- Prime complaint + late delivery

The current model forces these into a single intent.

---

## 10. What Is Misleading About My Headline Number?

The **41.5% accuracy** should not be interpreted as saying that the
complete support agent successfully resolves 41.5% of real-world
customer cases.

That number measures only **intent classification accuracy** on a
200-example golden evaluation set.

It does not directly measure:

- response helpfulness
- response correctness
- response-grounding quality
- escalation appropriateness
- actual customer issue resolution

There are additional limitations:

- the training labels are weakly generated
- the golden set is relatively small
- some categories have very few examples
- some messages contain multiple intents
- the classifier uses TF-IDF rather than semantic embeddings

For this reason, accuracy is reported together with macro F1,
baseline comparisons, retrieval analysis, escalation behavior, and
failure analysis.

---

## 11. Response-Quality Evaluation

A separate 30-example response evaluation sample is included.

Responses are scored on a 1–5 rubric:

| Score | Meaning |
|---:|---|
| 1 | Poor / irrelevant |
| 2 | Weak / only partially useful |
| 3 | Acceptable but incomplete |
| 4 | Good / useful |
| 5 | Excellent / directly useful |

The repository includes:

- `llm_judge.py` — reproducible LLM-as-judge harness
- `human_agreement.py` — human-rater agreement calculation

### LLM-as-judge

The judge evaluates:

- relevance
- helpfulness
- grounding
- appropriateness
- overall quality

No LLM-judge score is claimed in this repository because an API-key
backed evaluation was not executed during development.

### Human agreement

The repository includes a Cohen's kappa calculation for comparing two
independent human raters.

No agreement score is claimed unless two independent ratings are
actually available.

This avoids fabricating evaluation results.

---

## 12. UI

The project includes a Streamlit demonstration interface.

Run:

```bash
streamlit run app.py
```

The UI displays:

- customer message
- predicted intent
- model confidence
- response match score
- AUTO-HANDLE / ESCALATE decision
- explanation for the decision
- historically retrieved AmazonHelp response

The UI is a demonstration layer over the same classifier, retrieval,
and escalation logic used by the evaluation pipeline.

---

## 13. Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the benchmark:

```bash
python evaluate.py
```

Run the demo:

```bash
streamlit run app.py
```

The benchmark values shown in this README are generated by the committed
evaluation script and the committed training and golden datasets.

The 200-example golden set is never used for training.

---

## 14. Decision Log

See `DECISIONS.md` for the 15 non-obvious design decisions made during
development.

Key decisions include:

- AmazonHelp brand selection
- English-language filtering
- intent taxonomy design
- weak supervision
- balanced training data
- classifier choice
- macro-F1 evaluation
- retrieval-based response generation
- response-match threshold
- escalation policy
- high-risk intent escalation
- conservative automation
- UI design
- raw-dataset handling
- reproducibility choices

---

## 15. Limitations

This project is a research prototype rather than a production
customer-support system.

Important limitations include:

- weakly supervised training labels
- relatively small golden evaluation set
- rare intents with very few evaluation examples
- difficulty handling multi-intent messages
- lexical rather than semantic retrieval
- conservative automation coverage
- no live order or account-system integration
- no production authentication or access control

---

## 16. What I Would Improve With One More Week

### 1. Better training labels

Replace weak labels with substantially more human-labelled training
examples.

### 2. Semantic models

Use sentence embeddings or a stronger text encoder for intent
classification and historical-response retrieval.

### 3. Multi-intent classification

Allow a message to contain multiple intents instead of forcing one
category.

### 4. Intent-conditioned retrieval

Retrieve historical responses using both customer-message similarity and
predicted intent.

### 5. Better escalation calibration

Create a separate development set and tune thresholds based on the cost
of false automation versus unnecessary escalation.

### 6. Stronger response evaluation

Execute the LLM-as-judge harness and compare it against independent
human raters.

### 7. Privacy and safety

Add stronger PII detection and filtering before customer messages or
responses are stored or displayed.

### 8. End-to-end evaluation

Measure whether the suggested response actually solves the customer's
problem, rather than evaluating intent classification alone.

---

## 17. Repository Structure

```text
amazonhelp-support-agent/
│
├── README.md
├── DECISIONS.md
├── architecture.png
├── app.py
├── evaluate.py
├── llm_judge.py
├── human_agreement.py
├── requirements.txt
├── .gitignore
│
├── amazonhelp_golden_200.csv
├── amazonhelp_weak_training.csv
├── amazonhelp_response_pairs.csv
├── amazonhelp_response_eval_30.csv
└── amazonhelp_response_eval_30_to_rate.csv
```

The original raw `twcs.csv` dataset is intentionally excluded from the
repository.

---

## 18. Quick Start

```bash
pip install -r requirements.txt
python evaluate.py
streamlit run app.py
```

The core classifier, retrieval system, escalation policy, evaluation
pipeline, and demonstration UI can run locally without requiring an
external LLM API.