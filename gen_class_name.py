import os

root_dir = r"D:\2026\HK4\cs231\Project\demo\data\butterfly_dataset\test"

class_names = sorted([
    folder for folder in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, folder))
])

class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
idx_to_class = {idx : class_name for idx, class_name in enumerate(class_names)}
print(idx_to_class)

import json

with open("class_name.json", "w") as f:
    json.dump(idx_to_class, f)