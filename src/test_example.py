from playwright.sync_api import sync_playwright
from coordinate_automation import CoordinateAutomation

def test_example():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Initialize automation framework
        automation = CoordinateAutomation(page)
        
        try:
            # Navigate to the application
            automation.navigate("https://example.com")
            
            # Perform actions using labels from OCR
            automation.click("Login")
            automation.type("Username", "testuser")
            automation.type("Password", "testpass")
            automation.click("Submit")
            
            # Check for page changes and update coordinates
            automation.check_page_change()
            
            # Continue with more actions...
            automation.click("Dashboard")
            
        finally:
            browser.close()

if __name__ == "__main__":
    test_example() 