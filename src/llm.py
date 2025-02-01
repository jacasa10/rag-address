from transformers import pipeline

if __name__ == "__main__":
    classifier = pipeline("sentiment-analysis")

    print(classifier(["I've been waiting for a HuggingFace course my whole life."]))
