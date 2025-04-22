from behave.formatter.html import HTMLFormatter
import os
from datetime import datetime
import json

class CustomHTMLFormatter(HTMLFormatter):
    def __init__(self, stream_opener, config):
        super(CustomHTMLFormatter, self).__init__(stream_opener, config)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = "reports/html"
        self.screenshot_dir = os.path.join(self.report_dir, "screenshots")
        self.test_data = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features": [],
            "total_scenarios": 0,
            "passed_scenarios": 0,
            "failed_scenarios": 0,
            "skipped_scenarios": 0
        }
        
        # Create directories if they don't exist
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def feature(self, feature):
        self.feature = feature
        self.current_feature = {
            "name": feature.name,
            "description": feature.description,
            "scenarios": [],
            "total_steps": 0,
            "passed_steps": 0,
            "failed_steps": 0,
            "skipped_steps": 0
        }
        self.test_data["features"].append(self.current_feature)
        super(CustomHTMLFormatter, self).feature(feature)

    def scenario(self, scenario):
        self.scenario = scenario
        self.current_scenario = {
            "name": scenario.name,
            "steps": [],
            "status": "passed",
            "duration": 0
        }
        self.current_feature["scenarios"].append(self.current_scenario)
        self.test_data["total_scenarios"] += 1
        super(CustomHTMLFormatter, self).scenario(scenario)

    def step(self, step):
        self.step = step
        step_data = {
            "name": step.name,
            "keyword": step.keyword,
            "status": "pending",
            "duration": 0,
            "error_message": None,
            "screenshot": None
        }
        self.current_scenario["steps"].append(step_data)
        self.current_feature["total_steps"] += 1
        super(CustomHTMLFormatter, self).step(step)

    def result(self, step):
        # Take screenshot for failed steps
        if step.status == "failed" and hasattr(self, 'context'):
            try:
                screenshot_path = os.path.join(
                    self.screenshot_dir,
                    f"{self.feature.name}_{self.scenario.name}_{step.name}_{self.timestamp}.png"
                )
                self.context.page.screenshot(path=screenshot_path)
                step.screenshot = os.path.relpath(screenshot_path, self.report_dir)
            except Exception as e:
                print(f"Failed to take screenshot: {str(e)}")

        # Update test data
        step_data = self.current_scenario["steps"][-1]
        step_data["status"] = step.status
        step_data["duration"] = step.duration
        if step.status == "failed":
            step_data["error_message"] = step.error_message
            step_data["screenshot"] = step.screenshot
            self.current_scenario["status"] = "failed"
            self.current_feature["failed_steps"] += 1
            self.test_data["failed_scenarios"] += 1
        elif step.status == "passed":
            self.current_feature["passed_steps"] += 1
        elif step.status == "skipped":
            self.current_feature["skipped_steps"] += 1
            self.test_data["skipped_scenarios"] += 1

        if step.status == "passed" and self.current_scenario["status"] != "failed":
            self.current_scenario["status"] = "passed"
            self.test_data["passed_scenarios"] += 1

        super(CustomHTMLFormatter, self).result(step)

    def eof(self):
        # Add custom CSS and JavaScript to the report
        self.stream.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .summary {
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .feature {
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .scenario {
                    margin: 10px 0;
                    padding: 10px;
                    border-left: 4px solid #3498db;
                }
                .step {
                    margin: 5px 0;
                    padding: 5px;
                }
                .passed { color: #27ae60; }
                .failed { 
                    color: #e74c3c;
                    background-color: #ffdddd;
                }
                .skipped { color: #f39c12; }
                .screenshot {
                    max-width: 100%;
                    margin: 10px 0;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .error-message {
                    color: #e74c3c;
                    background-color: #ffdddd;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }
                .stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .stat-box {
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .stat-value {
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }
                .passed-value { color: #27ae60; }
                .failed-value { color: #e74c3c; }
                .skipped-value { color: #f39c12; }
                .duration {
                    color: #666;
                    font-size: 0.8em;
                }
                .toggle {
                    cursor: pointer;
                    color: #3498db;
                }
                .hidden {
                    display: none;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Test Report</h1>
                <p>Generated on: """ + self.test_data["start_time"] + """</p>
            </div>

            <div class="summary">
                <h2>Summary</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>Total Scenarios</h3>
                        <div class="stat-value">""" + str(self.test_data["total_scenarios"]) + """</div>
                    </div>
                    <div class="stat-box">
                        <h3>Passed</h3>
                        <div class="stat-value passed-value">""" + str(self.test_data["passed_scenarios"]) + """</div>
                    </div>
                    <div class="stat-box">
                        <h3>Failed</h3>
                        <div class="stat-value failed-value">""" + str(self.test_data["failed_scenarios"]) + """</div>
                    </div>
                    <div class="stat-box">
                        <h3>Skipped</h3>
                        <div class="stat-value skipped-value">""" + str(self.test_data["skipped_scenarios"]) + """</div>
                    </div>
                </div>
            </div>
        """)

        # Add features
        for feature in self.test_data["features"]:
            self.stream.write(f"""
            <div class="feature">
                <h2>{feature['name']}</h2>
                <p>{feature['description']}</p>
                <div class="stats">
                    <div class="stat-box">
                        <h3>Total Steps</h3>
                        <div class="stat-value">{feature['total_steps']}</div>
                    </div>
                    <div class="stat-box">
                        <h3>Passed</h3>
                        <div class="stat-value passed-value">{feature['passed_steps']}</div>
                    </div>
                    <div class="stat-box">
                        <h3>Failed</h3>
                        <div class="stat-value failed-value">{feature['failed_steps']}</div>
                    </div>
                    <div class="stat-box">
                        <h3>Skipped</h3>
                        <div class="stat-value skipped-value">{feature['skipped_steps']}</div>
                    </div>
                </div>
            """)

            # Add scenarios
            for scenario in feature['scenarios']:
                self.stream.write(f"""
                <div class="scenario">
                    <h3>{scenario['name']}</h3>
                    <div class="duration">Duration: {scenario['duration']:.2f}s</div>
                """)

                # Add steps
                for step in scenario['steps']:
                    self.stream.write(f"""
                    <div class="step {step['status']}">
                        <strong>{step['keyword']}</strong> {step['name']}
                        <div class="duration">Duration: {step['duration']:.2f}s</div>
                    """)
                    
                    if step['status'] == 'failed':
                        self.stream.write(f"""
                        <div class="error-message">
                            {step['error_message']}
                        </div>
                        """)
                        if step['screenshot']:
                            self.stream.write(f"""
                            <img src="{step['screenshot']}" class="screenshot" alt="Screenshot of failed step">
                            """)
                    
                    self.stream.write("</div>")
                
                self.stream.write("</div>")
            
            self.stream.write("</div>")

        # Add JavaScript for interactivity
        self.stream.write("""
            <script>
                function toggleSteps(element) {
                    const steps = element.nextElementSibling;
                    if (steps.style.display === "none") {
                        steps.style.display = "block";
                        element.textContent = "▼ Hide Steps";
                    } else {
                        steps.style.display = "none";
                        element.textContent = "▶ Show Steps";
                    }
                }
            </script>
        </body>
        </html>
        """)

        super(CustomHTMLFormatter, self).eof() 