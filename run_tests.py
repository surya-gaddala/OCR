import os
import sys
from behave.__main__ import main as behave_main
import shutil

def run_tests():
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    html_dir = os.path.join(reports_dir, "html")
    junit_dir = os.path.join(reports_dir, "junit")
    
    # Clean up old reports
    if os.path.exists(reports_dir):
        shutil.rmtree(reports_dir)
    
    # Create fresh directories
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(junit_dir, exist_ok=True)
    
    # Run behave with custom formatter
    sys.argv = [
        "behave",
        "--format", "pretty",
        "--format", "html",
        "--format", "junit",
        "--format", "src.custom_html_formatter:CustomHTMLFormatter",
        "--outfile", os.path.join(reports_dir, "behave_results.txt"),
        "--junit-directory", junit_dir,
        "--html-directory", html_dir,
        "features"
    ]
    
    # Run the tests
    result = behave_main()
    
    # Print report location
    print("\nReports generated:")
    print(f"HTML Report: {os.path.abspath(os.path.join(html_dir, 'index.html'))}")
    print(f"JUnit Report: {os.path.abspath(junit_dir)}")
    print(f"Text Report: {os.path.abspath(os.path.join(reports_dir, 'behave_results.txt'))}")
    
    return result

if __name__ == "__main__":
    run_tests() 