from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
import mysql.connector, joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
import os
import base64
from io import BytesIO
from datetime import datetime

# Set non-interactive Matplotlib backend to avoid GUI warnings
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'book' 

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='book'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']

        if password == c_password:
            query = "SELECT email FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])

            if email not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)

                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        query = "SELECT email FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email in email_data_list:
            query = "SELECT * FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password == password__data[0][3]:
                session["user_email"] = email
                session["user_id"] = password__data[0][0]
                session["user_name"] = password__data[0][1]

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

########################################################################################################################
############################################## PREDICTION SECTION #####################################################
########################################################################################################################

# Load the trained model
model = joblib.load(r"models\Random_Forest.pkl")

# Create SHAP explainer once
explainer = shap.TreeExplainer(model)

# Labels for prediction result
prediction_label = {
    1: "This person has a chance of canceling the booking.",
    0: "This person doesn't have a chance of canceling the booking."
}

# Feature names for SHAP plots
feature_names = [
    'LeadTime', 'ArrivalDateYear', 'ArrivalDateMonth', 'ArrivalDateWeekNumber',
    'ArrivalDateDayOfMonth', 'StaysInWeekendNights', 'StaysInWeekNights', 'Adults',
    'Children', 'Babies', 'Meal', 'Country', 'MarketSegment', 'DistributionChannel',
    'IsRepeatedGuest', 'PreviousCancellations', 'PreviousBookingsNotCanceled',
    'ReservedRoomType', 'AssignedRoomType', 'BookingChanges', 'DepositType',
    'DaysInWaitingList', 'CustomerType', 'ADR', 'RequiredCarParkingSpaces',
    'TotalOfSpecialRequests'
]

# Mapping dictionaries
Meal_label = {
    'BB': 0, 'FB': 1, 'HB': 2, 'SC': 3
}
Country_label = {
    'France': 0, 'Germany': 1, 'Portugal': 2, 'Spain': 3
}
MarketSegment_label = {
    'Groups': 0, 'Offline TA/TO': 1, 'Online TA': 2
}
DistributionChannel_label = {
    'Corporate': 0, 'Direct': 1, 'TA': 2, 'TO': 3
}
ReservedRoomType_label = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4
}
AssignedRoomType_label = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4
}
DepositType_label = {
    'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2
}
CustomerType_label = {
    'Contract': 0, 'Transient': 1, 'Transient-Party': 2
}

def plot_to_base64():
    """Convert matplotlib plot to base64 string for HTML embedding"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic

# --- Prediction function with SHAP XAI ---
def prediction_func(input_data_raw):
    input_data = input_data_raw.copy()

    # Map categorical fields
    input_data[10] = Meal_label[input_data[10]]
    input_data[11] = Country_label[input_data[11]]
    input_data[12] = MarketSegment_label[input_data[12]]
    input_data[13] = DistributionChannel_label[input_data[13]]
    input_data[17] = ReservedRoomType_label[input_data[17]]
    input_data[18] = AssignedRoomType_label[input_data[18]]
    input_data[20] = DepositType_label[input_data[20]]
    input_data[22] = CustomerType_label[input_data[22]]

    # Convert input into NumPy array (1 sample, n_features)
    input_array = np.array(input_data).reshape(1, -1)

    # Prediction
    prediction = model.predict(input_array)
    result = prediction_label[prediction[0]]
    print("\nPrediction:", result)

    # --- SHAP XAI Visualization ---
    shap_values = explainer.shap_values(input_array)

    # Check if output is list or ndarray
    if isinstance(shap_values, list):
        shap_value_to_plot = shap_values[1][0]
        expected_value_to_plot = explainer.expected_value[1]
    else:
        shap_value_to_plot = shap_values[0, :, 1]
        expected_value_to_plot = explainer.expected_value[1]

    # Create plots
    plots = {}
    
    # 1. Waterfall plot
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_value_to_plot,
            base_values=expected_value_to_plot,
            data=input_array[0],
            feature_names=feature_names
        ),
        show=False
    )
    plots['waterfall_plot'] = plot_to_base64()
    plt.close()

    # 2. Bar plot (feature importance)
    plt.figure(figsize=(10, 8))
    shap.plots.bar(
        shap.Explanation(
            values=shap_value_to_plot,
            base_values=expected_value_to_plot,
            feature_names=feature_names
        ),
        show=False
    )
    plots['bar_plot'] = plot_to_base64()
    plt.close()

    return result, plots

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            LeadTime = float(request.form['LeadTime'])
            
            # Handle the new date input
            arrival_date_str = request.form['ArrivalDate']
            arrival_date = datetime.strptime(arrival_date_str, '%Y-%m-%d')
            ArrivalDateYear = float(arrival_date.year)
            ArrivalDateMonth = float(arrival_date.month)
            ArrivalDateDayOfMonth = float(arrival_date.day)
            
            # Calculate week number
            ArrivalDateWeekNumber = float(arrival_date.isocalendar()[1])
            
            StaysInWeekendNights = float(request.form['StaysInWeekendNights'])
            StaysInWeekNights = float(request.form['StaysInWeekNights'])
            Adults = float(request.form['Adults'])
            Children = float(request.form.get('Children', 0))
            Babies = float(request.form.get('Babies', 0))
            Meal = request.form['Meal']
            Country = request.form['Country']
            MarketSegment = request.form['MarketSegment']
            DistributionChannel = request.form['DistributionChannel']
            IsRepeatedGuest = float(request.form['IsRepeatedGuest'])
            
            # Add default values for fields not in the simplified form
            PreviousCancellations = 0.0
            PreviousBookingsNotCanceled = 0.0
            ReservedRoomType = "A"
            AssignedRoomType = "A"
            BookingChanges = 0.0
            DepositType = request.form['DepositType']
            DaysInWaitingList = 0.0
            CustomerType = "Transient"
            ADR = float(request.form['ADR'])
            RequiredCarParkingSpaces = 0.0
            TotalOfSpecialRequests = float(request.form.get('TotalOfSpecialRequests', 0))

            input_data = [
                LeadTime, ArrivalDateYear, ArrivalDateMonth, ArrivalDateWeekNumber,
                ArrivalDateDayOfMonth, StaysInWeekendNights, StaysInWeekNights, Adults,
                Children, Babies, Meal, Country, MarketSegment, DistributionChannel,
                IsRepeatedGuest, PreviousCancellations, PreviousBookingsNotCanceled,
                ReservedRoomType, AssignedRoomType, BookingChanges, DepositType,
                DaysInWaitingList, CustomerType, ADR, RequiredCarParkingSpaces,
                TotalOfSpecialRequests
            ]

            result, shap_plots = prediction_func(input_data)

            return render_template('prediction.html', result=result, shap_plots=shap_plots)
        
        except Exception as e:
            flash(f"Error processing your request: {str(e)}", "error")
            return render_template('prediction.html')
    
    return render_template('prediction.html')
    
    return render_template('prediction.html')
if __name__ == '__main__':
    app.run(debug=True)