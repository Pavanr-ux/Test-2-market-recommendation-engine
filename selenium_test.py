from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Initialize WebDriver
driver = webdriver.Chrome()  # Or use another WebDriver if you're using a different browser

# Start by opening the Flask app's homepage
driver.get("http://127.0.0.1:5000")  # Make sure your Flask app is running on this URL

# Allow the page to load
time.sleep(2)

# Locate the form fields and submit button
unit_price_field = driver.find_element(By.ID, "UnitPrice")
quantity_field = driver.find_element(By.ID, "Quantity")
country_field = driver.find_element(By.ID, "Country")
submit_button = driver.find_element(By.XPATH, "//button[@type='submit']")

# Fill in the form
unit_price_field.send_keys("10.5")
quantity_field.send_keys("5")
country_field.send_keys("1")  # Assuming '1' corresponds to a valid country code

# Submit the form
submit_button.click()

# Wait for the prediction result to load
time.sleep(2)  # Adjust this if the prediction response takes longer

# Check if prediction result is displayed
try:
    result_text = driver.find_element(By.ID, "result").text
    print("Test Passed: Prediction result displayed.")
    print("Result:", result_text)
except:
    print("Test Failed: Prediction result not displayed.")

# Close the browser
driver.quit()
