The application provides a dashboard to climate activists who are interested in the environmental challenges of the San Francisco Bay area where water quality measurements data can be discovered interactively. 

***Web application***
If you don't want to run the application locally, use the following link to an online web-enabled visualization:
https://bognarek-dash.herokuapp.com/ in your browser.

***Running locally - requirements***
To run the application Phyton 3 should be installed. 

To install all of the required Phyton packages, simply run:
pip install -r requirements.txt

***How to use the app***:
Run this app locally by:
python app.py
Open http://127.0.0.1:8050/ in your browser.

Use the dropdown boxes and sliders to select measurements (max. 2 at the same time) and time or depth ranges. Select data points from the map to visualize data for the selected station(s). Selection could be done by clicking on individual data points or using the lasso tool to capture multiple data points. Use the graph's menu bar to select, zoom or save the plots. 