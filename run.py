import sys

if len(sys.argv) < 2:
    print("Usage: python run.py train")
    sys.exit(1)

if sys.argv[1].lower() == "train":
    import src.train
else:
    print("Unknown command.")
