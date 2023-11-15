from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import urllib.request
import threading

def download_image(image_url, image_path):
    try:
        urllib.request.urlretrieve(image_url, image_path)
        print(f"Image {image_path} downloaded.")
    except Exception as e:
        print(f"Error downloading image: {e}")

# Set up Selenium webdriver
driver = webdriver.Chrome()

# Create a folder to save the images
folder_name = "Termites"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Open the webpa
url = "https://bugguide.net/index.php?q=search&keys=termites&edit%5Btype%5D%5Bbgimage%5D=on"
driver.get(url)

# Explicitly wait for the images to load
wait = WebDriverWait(driver, 2)
image_elements = wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "img")))

# Create a list to store threads
threads = []

# Download and save the images using threads
for i, image_element in enumerate(image_elements):
    image_url = image_element.get_attribute("src")
    if image_url and image_url.startswith("http"):
        image_name = f"image_{i}.jpg"
        image_path = os.path.join(folder_name, image_name)
        thread = threading.Thread(target=download_image, args=(image_url, image_path))
        threads.append(thread)
        thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Close the browser
driver.quit()
