# AmazonHelp AI Customer Support Agent

An AI customer-support agent built on the Customer Support on Twitter
dataset, focused on the AmazonHelp brand.

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

---

## 2. Architecture

```text
Customer message
       |
       v
 TF-IDF + Logistic Regression
       |
       v
 Intent + confidence
       |
       +----------------------+
       |                      |
       v                      v
Historical response      Risk / threshold
retrieval                    checks
       |                      |
       +----------+-----------+
                  |
                  v
        AUTO-HANDLE / ESCALATE
                  |
                  v
        Suggested response