# AI Missing Person Detection System

An AI-powered surveillance system that helps detect missing persons using facial recognition and sends real-time alerts.

## Features

- Register missing persons with photo and details
- Store cases using SQLite database
- Real-time face detection using OpenCV
- Automatic recognition of registered persons
- WhatsApp alert system using Twilio API
- Case management dashboard
- REST APIs for retrieving case data

## Technologies Used

- Python
- Flask
- OpenCV
- SQLite
- Twilio WhatsApp API
- HTML / CSS

## System Architecture

Register Missing Person  
↓  
Store image and data in database  
↓  
Camera surveillance scans faces  
↓  
Face recognition compares with registered faces  
↓  
Match detected  
↓  
WhatsApp alert sent to registered phone number  

## Project Structure
MissingPersonSystem
│
├── app.py
├── missing_persons.db
│
├── templates
│ ├── home.html
│ ├── register.html
│ ├── surveillance.html
│ ├── cases.html
│ ├── edit_case.html
│ └── match.html
│
├── static
│ └── person_images

## Installation

Clone the repository:
https://github.com/Subramanian-S7/Missing-person-detection-using-AI

Install dependencies:
pip install flask opencv-python numpy twilio reportlab.

Run the application:
python app.py

Open in browser:
http://127.0.0.1:5000

## Future Improvements

- Live CCTV integration
- Police monitoring dashboard
- Google Maps location tracking
- Deep learning face recognition model
- Cloud deployment

## Author
Subramanian S

