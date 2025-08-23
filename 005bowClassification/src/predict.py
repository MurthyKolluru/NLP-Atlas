import argparse, joblib, sys
from utils import load_target_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to classify")
    args = parser.parse_args()

    if not args.text:
        print("Pass text via --text \"your sentence here\"")
        sys.exit(1)

    pipe = joblib.load("models/clf.joblib")
    target_names = load_target_names()
    pred = pipe.predict([args.text])[0]
    proba = None
    # RandomForest supports predict_proba only if configured; many sklearn forests do.
    try:
        proba = pipe.predict_proba([args.text])[0]
    except Exception:
        pass

    print(f"Predicted class: {target_names[pred]}")
    if proba is not None:
        # show top-3 probabilities if available
        top = sorted(list(enumerate(proba)), key=lambda x: x[1], reverse=True)[:3]
        for idx, p in top:
            print(f"  {target_names[idx]:20s} {p:.3f}")

if __name__ == "__main__":
    main()
