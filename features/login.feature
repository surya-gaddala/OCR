Feature: Coordinate-Based Form Automation using OCR

  Scenario: Successful login using coordinates detected by OCR
    Given I navigate to "https://tutorialsninja.com/demo/index.php?route=account/login" and perform OCR
    # Then I see the page title as "My Account"
    When I enter "suryaiiit.517@gmail.com" using label "E-Mail Address"
    And I enter "Test@123" using label "Password"
    And I click using label "Login" and perform OCR
    And I click using label "Tablets" and perform OCR
    And I click using label "Samsung Galaxy Tab 10.1" and perform OCR
    # And I click using label "Add to Cart" and perform OCR
    # And I click using label "Shopping Cart" and perform OCR
    # And I click using label "Checkout" and perform OCR
    # And I click using label "I agree to the terms and conditions" and perform OCR
    

# Feature: Coordinate-Based Form Automation using OCR

#   Scenario: Successful login using coordinates detected by OCR
#     Given I navigate to "https://demowebshop.tricentis.com/login" and perform OCR
#     # Then I see the page title as "My Account"
#     When I enter "gaddala.surya@dragonflytest.com" using label "Email"
#     And I enter "Test@123" using label "Password"
#     And I click using label "Login" and perform OCR
#     And I wait for 10 seconds
#     And I see the label "Newsletter"
#     And I click using label "Books" and perform OCR
#     And I click using label "Computing and Internet"
#     And I click using label "Add to cart" 
#     And I click using label "Shopping cart" and perform OCR
#     When I check the checkbox using label "Computing and internet"


# Feature: Coordinate-Based Form Automation using OCR

#   Scenario: Successful login using coordinates detected by OCR
#     Given I navigate to "https://hab.instarresearch.com/wix/56789/p159939540878.aspx?QClink=1" and perform OCR
#     # Then I see the page title as "My Account"
#     When I enter "1" using label "BG_Target"
#     And I enter "1" using label "BG_DSE_Segment_Code"
#     And I click using label "Next"
#     And I wait for 10 seconds
#     And I see the label "QID: TargetH"
#     And I click using label "Next"
#     # And I click using label "Next"
#     # # And I click using label "1_HAB"
#     # # And I click using label "Next"
#     # # And I click using label "United Arab Emirates"


# Feature: User form interactions using OCR-based automation

#   Background:
#     Given I navigate to "http://testautomationpractice.blogspot.com/2018/09/automation-form.html" and perform OCR
#     And I wait for 5 seconds for the page to load

#   Scenario: Successful form filling and submission
#     When I enter "surya" using label "Enter Name"
#     And I enter "surya@gmail.com" using label "Enter EMail"
#     And I enter "1234567890" using label "Enter Phone"
#     # And I check the checkbox using label "Check me out if you Love IceCreams!"
#     And I select the radio button using label "Male"
#     And I slide the slider labeled "Slider" to position 50
#     And I drag element labeled "Drag me to my target" and drop on "Drop here"
#     # And I click using label "Submit"
#     # Then I wait for 10 seconds
#     # And I see the label "Form submitted successfully"
