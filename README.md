# RasperryPi-Vibration-Monitor


The rasperrypi vibration monitor that forwards the vibrations to the autoencoder and middleware if an anomaly is the signal is detected.

This should be sat up as the last step as this requires the middleware, the api and app client. 

set up the .env file as in .env.example. The unit (raspberry pi), should be registered in the database, if you have not already done so. This is to identify the unit to know where the anomaly came from. 

# Install 

- sudo apt-get install build-essential python-dev
- sudo apt-get install libatlas-base-dev

# Upload the code to raspberry pi and run:

```python3 app.py``` to start the application.