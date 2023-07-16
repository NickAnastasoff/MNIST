import os
import random

def make_test_set(train_dir, test_dir):
  """Makes a test set from the given train set.

  Args:
    train_dir: The directory containing the train set.
    test_dir: The directory to store the test set.
  """

  if not os.path.exists(test_dir):
    os.mkdir(test_dir)

  for digit in range(10):
    digit_dir = os.path.join(train_dir, str(digit))
    test_digit_dir = os.path.join(test_dir, str(digit))
    if not os.path.exists(test_digit_dir):
      os.mkdir(test_digit_dir)

    files = os.listdir(digit_dir)
    for i, file in enumerate(files):
      if i % 5 == 0:
        random_index = random.randint(0, len(files) - 1)
        file_to_move = files[random_index]

        os.system(f"cp {os.path.join(digit_dir, file_to_move)} {os.path.join(test_digit_dir, file_to_move)}")
        print(f"Moved {file_to_move} to {test_digit_dir}")

if __name__ == "__main__":
  train_dir = "data/train"
  test_dir = "data/test"
  make_test_set(train_dir, test_dir)
  print("Done!")