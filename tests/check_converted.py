import os

converted_dir = "converted"
files = os.listdir(converted_dir)
print(f"converted klasöründe {len(files)} dosya var:")
for f in files[:10]:
    print(f)
