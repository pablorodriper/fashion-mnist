import subprocess

def download_with_curl(url, path):
    """
    Download a file from a URL to a path using curl.
    """
    curl_cmd = ['curl', url, '-o', path]
    subprocess.call(curl_cmd)

train_images_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
train_images_path = "../data/original/train-images-idx3-ubyte.gz"

train_labels_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
train_labels_path = "../data/original/train-labels-idx1-ubyte.gz"

test_images_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
test_images_path = "../data/original/t10k-images-idx3-ubyte.gz"

test_labels_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
test_labels_path = "../data/original/t10k-labels-idx1-ubyte.gz"

download_with_curl(train_images_url, train_images_path)
download_with_curl(train_labels_url, train_labels_path)
download_with_curl(test_images_url, test_images_path)
download_with_curl(test_labels_url, test_labels_path)

print("Download complete.")