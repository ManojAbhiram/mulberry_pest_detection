import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# Initialize the Chrome driver
driver = webdriver.Chrome()

# Define the search query
search_query = "Mulberry Whitefly"

# Create a directory to save the images
if not os.path.exists(search_query):
    os.mkdir(search_query)

# Function to download images from a single page
def download_images_from_page(page_number):
    # Calculate the number of scrolls needed to simulate "Show more results"
    scroll_count = 10

    # Construct the search URL
    search_url = f"https://bugguide.net/index.php?q=search&keys=mulberry+whitefly&edit%5Btype%5D%5Bbgimage%5D=on"

    # Open the search URL
    driver.get(search_url)

    # Scroll to load more images
    for _ in range(scroll_count):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Give some time for new images to load

    # Extract image URLs from the search results
    image_elements = driver.find_elements("css selector", ".rg_i")

    # Extract the image URLs using data-src attributes
    image_urls = []
    for element in image_elements:
        data_src = element.get_attribute("data-src")
        if data_src:
            image_urls.append(data_src)

    # Download and save the images
    for i, url in enumerate(image_urls):
        response = requests.get(url)
        if response.status_code == 200:
            # Determine the file extension based on Content-Type header
            content_type = response.headers.get("Content-Type")
            if content_type:
                image_extension = content_type.split("/")[-1]
                if '?' in image_extension:
                    image_extension = image_extension.split('?')[0]
            else:
                image_extension = "jpg"  # Default to JPG if Content-Type is not available

            image_name = f"{search_query}_page{page_number}_{i+1}.{image_extension}"
            image_name = "".join([c for c in image_name if c.isalnum() or c in ['.', '_']])  # Remove invalid characters
            image_path = os.path.join(search_query, image_name)

            with open(image_path, "wb") as f:
                f.write(response.content)
                print(f"Downloaded {image_path}")
        else:
            print(f"Failed to download image {i+1}")

# Download images from multiple pages
total_pages = 1  # Change this to the number of pages you want to download from
for page_number in range(1, total_pages + 1):
    download_images_from_page(page_number)

# Close the browser when done
driver.quit()
