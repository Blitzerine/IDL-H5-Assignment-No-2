import os

# Root = folder where this script is saved
root = os.path.dirname(os.path.abspath(__file__))

print("Root path is:", root)

folders = [
    os.path.join(root, "src"),
    os.path.join(root, "outputs"),
    os.path.join(root, "report"),
]

files = [
    os.path.join(root, "src", "preprocess_mnist.py"),
    os.path.join(root, "outputs", ".gitignore"),
    os.path.join(root, "report", "comparison_table.md"),
    os.path.join(root, "README.md"),
    os.path.join(root, "requirements.txt"),
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for f in files:
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as fp:
            fp.write("")

print("Created folders and files inside:")
print(os.listdir(root))
