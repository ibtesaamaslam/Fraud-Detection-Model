# ---------------------------------------------
# HUGGING FACE AI EXPLANATION MODULE
# ---------------------------------------------
# This file generates human-like explanations
# for fraud predictions using a transformer model
# ---------------------------------------------

from transformers import pipeline
import textwrap

# ---------------------------------------------
# LOAD MODEL (RUNS ONLY ONCE)
# ---------------------------------------------
# FLAN-T5 is MUCH better than GPT-2 for reasoning tasks
# It is trained to follow instructions and explain things

try:
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )
    MODEL_NAME = "FLAN-T5"
except Exception as e:
    generator = None
    MODEL_NAME = "Fallback"
    print("⚠️ Hugging Face model failed to load:", e)


# ---------------------------------------------
# HELPER: CLEAN TEXT
# ---------------------------------------------
def clean_output(text):
    """
    Makes output clean and readable
    """
    text = text.strip()
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    return text


# ---------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------
def generate_explanation(transaction, prediction, probability):
    """
    Generates a human-readable explanation of fraud detection

    Parameters:
    - transaction: pandas Series (one row)
    - prediction: 0 (safe) or 1 (fraud)
    - probability: float (0 → 1)

    Returns:
    - string explanation
    """

    # -----------------------------------------
    # SAFETY CHECK (IF MODEL NOT LOADED)
    # -----------------------------------------
    if generator is None:
        return (
            f"Prediction: {'Fraud' if prediction == 1 else 'Safe'}.\n"
            f"Confidence: {round(probability * 100, 2)}%.\n"
            "AI explanation is unavailable (model not loaded)."
        )

    # -----------------------------------------
    # LIMIT FEATURES (avoid overload)
    # -----------------------------------------
    # We only pass top features (first 15) to keep prompt clean
    txn_dict = transaction.to_dict()
    limited_features = dict(list(txn_dict.items())[:15])

    # -----------------------------------------
    # BUILD PROMPT (THIS IS CRITICAL)
    # -----------------------------------------
    prompt = f"""
    You are a financial fraud analyst.

    A transaction has been analyzed by an AI system.

    Prediction: {"Fraud" if prediction == 1 else "Safe"}
    Confidence: {round(probability * 100, 2)}%

    Key transaction features:
    {limited_features}

    Explain in simple, human terms:
    - Why this transaction is risky or safe
    - What signals might indicate fraud
    - Keep it short and clear (4-5 lines)
    """

    # Clean formatting
    prompt = textwrap.dedent(prompt)

    # -----------------------------------------
    # GENERATE RESPONSE
    # -----------------------------------------
    try:
        result = generator(prompt, max_length=200, do_sample=True)

        explanation = result[0]["generated_text"]
        explanation = clean_output(explanation)

        return explanation

    except Exception as e:
        return (
            f"Prediction: {'Fraud' if prediction == 1 else 'Safe'}.\n"
            f"Confidence: {round(probability * 100, 2)}%.\n"
            f"AI explanation failed: {str(e)}"
        )