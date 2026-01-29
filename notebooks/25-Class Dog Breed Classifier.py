import os
import random
import shutil
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================
# 🔧 PATHS (EDIT ONCE)
# ============================
PROJECT_ROOT = Path(
    "C:/Users/fobio/Documents/Msc Game Dev Project/"
    "Machine Learing & Deep Learining/"
    "CI7521_CW2_Group22/CI7521_CW2_Group22/"
    "pet-breed-classification-deep-learning"
)

IMAGES_DIR = PROJECT_ROOT / "datasets/images"
ANNOTATIONS_DIR = PROJECT_ROOT / "datasets/annotations"
OUTPUT_DIR = PROJECT_ROOT / "datasets/Dog_Breed"

# ============================
# 🔍 SANITY CHECK
# ============================
def print_directories():
    print("\n📁 DIRECTORY CHECK")
    print("Images dir:", IMAGES_DIR)
    print("Annotations dir:", ANNOTATIONS_DIR)
    print("Images found:", len(list(IMAGES_DIR.glob("*.jpg"))))
    print("list.txt exists:", (ANNOTATIONS_DIR / "list.txt").exists())
    print("-" * 40)

# ============================
# 📝 CREATE list.csv
# ============================
def create_list_csv():
    list_txt = ANNOTATIONS_DIR / "list.txt"
    out_csv = ANNOTATIONS_DIR / "list.csv"

    rows = []
    with open(list_txt, "r") as f:
        lines = f.readlines()[6:]
        for line in lines:
            fname, class_id, species, breed = line.strip().split()
            rows.append([fname, int(class_id), int(species), int(breed)])

    df = pd.DataFrame(
        rows,
        columns=["Filename", "Class ID", "Species", "Breed ID"]
    )
    df.to_csv(out_csv, index=False)
    print("✅ list.csv created:", out_csv)

# ============================
# 🐶 CREATE DOG DATASET
# ============================
def create_dog_breed_dirs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    df = pd.read_csv(ANNOTATIONS_DIR / "list.csv")

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    dog_breeds = df[df["Species"] == 2]["Breed ID"].unique()

    def split_and_copy(images, breed):
        train_split = int(0.8 * len(images))
        val_split = int(0.9 * len(images))

        splits = {
            "train": images[:train_split],
            "val": images[train_split:val_split],
            "test": images[val_split:]
        }

        for split_name, files in splits.items():
            breed_dir = OUTPUT_DIR / split_name / str(breed)
            breed_dir.mkdir(parents=True, exist_ok=True)

            for fname in files:
                src = IMAGES_DIR / f"{fname}.jpg"
                dst = breed_dir / f"{fname}.jpg"
                if src.exists():
                    shutil.copy(src, dst)

    for breed in dog_breeds:
        images = df[
            (df["Species"] == 2) & (df["Breed ID"] == breed)
        ]["Filename"].tolist()

        random.shuffle(images)
        split_and_copy(images, breed)

    print("✅ Dog_Breed dataset created")

    return (
        OUTPUT_DIR / "train",
        OUTPUT_DIR / "val",
        OUTPUT_DIR / "test"
    )

# ============================
# 📦 LOAD DATASETS
# ============================
def load_dataset_from_directory(
    train_dir, val_dir, test_dir,
    batch_size, img_height, img_width
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode="categorical"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        label_mode="categorical"
    )

    # ✅ capture before prefetch
    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, num_classes

# ============================
# 🧹 CLEAN CORRUPT IMAGES
# ============================
def clean_corrupt_images(root_dir):
    removed = 0
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                try:
                    Image.open(os.path.join(root, f)).verify()
                except:
                    os.remove(os.path.join(root, f))
                    removed += 1
    print(f"Removed {removed} corrupt images")

# ============================
# 🚀 RUN
# ============================
def main():
    print_directories()

    create_list_csv()
    train_dir, val_dir, test_dir = create_dog_breed_dirs()
    clean_corrupt_images(OUTPUT_DIR)

    batch_size = 64
    img_height = 224
    img_width = 224

    train_ds, val_ds, test_ds, num_classes = load_dataset_from_directory(
        train_dir, val_dir, test_dir,
        batch_size, img_height, img_width
    )

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False

    model = Sequential([
        layers.Lambda(preprocess_input),   # ✅ INPUT FIXED HERE
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50
    )

    model.evaluate(test_ds)
    model.save("dog_breed_classifier_model.h5")
    print("🎉 MODEL SAVED: dog_breed_classifier_model.h5")

if __name__ == "__main__":
    main()
