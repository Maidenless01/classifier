import os
import tensorflow as tf

def clean_dataset(dataset_dir):
    """Remove images that TensorFlow's own decoder cannot load.
    PIL is too lenient with JPEG corruption; using TF ensures we catch
    every file that would crash training (e.g. 'extraneous bytes before
    marker 0xd9' / mismatched row-size errors)."""
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                raw = tf.io.read_file(fpath)
                # channels=3 matches what image_dataset_from_directory uses
                tf.image.decode_image(raw, channels=3, expand_animations=False)
            except Exception as e:
                print(f"Removing bad image {fpath}: {e}")
                os.remove(fpath)
                num_skipped += 1
    print(f"Deleted {num_skipped} images")

if __name__ == "__main__":
    clean_dataset(r"c:\Users\ASUS\Desktop\classifier\PetImages")
