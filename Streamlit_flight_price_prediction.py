import streamlit as st
import pandas as pd
import pickle 
import mlflow
import datetime

import seaborn as sns
import matplotlib.pyplot as plt

flight_model = pickle.load(open("C:/Users/Admin/Downloads/xgboost_model.pkl", "rb"))
satisfaction_model = pickle.load(open("C:/Users/Admin/Downloads/best_model_xgbclassifier.pkl", "rb"))

customer_data = pd.read_csv("C:/Users/Admin/OneDrive/Documents/Flight ML project/Passenger_Satisfaction_csv.csv")

flight_data = pd.read_csv("C:/Users/Admin/OneDrive/Documents/Flight ML project/Flight_Price_csv.csv")

flight_eda = pd.read_csv("C:/Users/Admin/OneDrive/Documents/Flight ML project/flight_eda.csv")
customer_eda = pd.read_csv("C:/Users/Admin/OneDrive/Documents/Flight ML project/customer_eda.csv")



mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Flight & Customer Satisfaction Prediction")


st.set_page_config(page_title="Flight & Customer Satisfaction Prediction")

st.sidebar.title("Navigate")
choice = st.sidebar.radio(label="Select a section",options=["Home","Flight Price Prediction", "Customer Satisfaction Prediction", "Data Visualization"])


if choice == "Home" :
    st.title("Flight Price & Customer Satisfaction Prediction")
    st.write("This page shows the analysis of the Flight price prediction and Customer staisfaction prediction")


elif choice == "Flight Price Prediction":
    st.title("Flight Price Prediction")
    # if all(col in flight_data.columns for col in ["Year", "Month", "Day"]):
    #     # Combine Year, Month, and Day into a single date column
    #     flight_data["Date_of_Journey"] = pd.to_datetime(flight_data[[flight_data["Year"], flight_data["Month"], flight_data["Day"]]])
        
        
    #     unique_dates = flight_data["Date_of_Journey"].dt.strftime("%Y-%m-%d").unique()

    
    # User input fields
    airline = st.selectbox("Airline", ["Airline_Air India",	"Airline_GoAir","Airline_IndiGo","Airline_Jet Airways","Airline_Jet Airways Business","Airline_Multiple carriers",
                                       "Airline_Multiple carriers Premium economy","Airline_SpiceJet",	"Airline_Trujet","Airline_Vistara",	"Airline_Vistara Premium economy", "Airline_Air Asia"]
                                       )     
    source = st.selectbox("Source", ["Source_Chennai","Source_Delhi","Source_Kolkata","Source_Mumbai", "Source_Banglore"])  
    destination = st.selectbox("Destination", ["Destination_Cochin","Destination_Delhi","Destination_Hyderabad","Destination_Kolkata","Destination_New Delhi","Destination_Banglore"])  
    # time_options = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]
    # departure_time = st.selectbox("Departure Time", time_options)
    # arrival_time = st.selectbox("Arrival Time", time_options)
    default_date = datetime.date.today()

    # Calendar input
    date_of_journey = st.date_input("Select Date of Journey", default_date)

    # Display selected date
    # st.write("Selected Date:", date_of_journey)
    total_stops = st.slider("Total Stops", 0, 4, 2, 1)
    duration_hours = st.number_input("Duration Hours", min_value=0, max_value=24, step=1)
    duration_minutes = st.number_input("Duration Minutes", min_value=0, max_value=59, step=1)
    # price_per_minute = st.number_input("Price Per Minute",min_value=0, max_value=4000, step = 50)

    # departure_hour, departure_minute = map(int, departure_time.split(":"))
    # arrival_hour, arrival_minute = map(int, arrival_time.split(":"))


      

    if st.button("Predict Price"):
        with mlflow.start_run():

            model_features = flight_model.get_booster().feature_names
            
            
            input_data = pd.DataFrame([[ date_of_journey, total_stops, duration_hours, duration_minutes]],
                                columns=["Date of Journey","Total_Stops", "Duration_Hours", "Duration_Min"]
                            )


            for col in model_features:
                if col not in input_data.columns:
                    input_data[col] = 0  # Add missing categorical features

       
            # Reorder columns to match model's expected input
            input_data = input_data[model_features]
            input_data = input_data.astype(float)


            predicted_price = flight_model.predict(input_data)[0]
            mlflow.log_params(input_data.to_dict(orient='records')[0])
            mlflow.log_metric("Predicted Price", predicted_price)
            st.success(f"Estimated Flight Price: {predicted_price:.2f}")


elif choice == "Customer Satisfaction Prediction":
    st.title("Customer Satisfaction Prediction")

    # User input fields
    # gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Disloyal", "Loyal"])
    # age = st.number_input("Age", min_value=10, max_value=100, step=1)
    travel_type = st.selectbox("Type of Travel", ["Personal", "Business"])
    flight_class = st.selectbox("Class", ["Economy", "Business", "First Class"])
    # flight_distance = st.number_input("Flight Distance", min_value=100, max_value=10000, step=10)

    # Review-related columns
    inflight_wifi = st.slider("Inflight wiFi service", 0, 5, 3)
    # ease_booking = st.slider("Ease of Online booking", 0, 5, 3)
    # gate_location = st.slider("Gate location", 0, 5, 3)
    # food_drink = st.slider("Food and drink", 0, 5, 3)
    online_boarding = st.slider("Online boarding", 0, 5, 3)
    seat_comfort = st.slider("Seat comfort", 0, 5, 3)
    inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)
    on_board_service = st.slider("On-board service", 0, 5, 3)
    leg_room_service = st.slider("Leg room service", 0, 5, 3)
    # baggage_handling = st.slider("Baggage handling", 0, 5, 3)
    checkin_service = st.slider("Checkin service", 0, 5, 3)
    # inflight_service = st.slider("Inflight service", 0, 5, 3)  
    # departure_arrival_convenient = st.slider("Departure/Arrival time convenient", 0, 5, 3)  
    # departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, max_value=1000, step=1)
    # arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, max_value=1000, step=1)
    # cleanliness = st.slider("Cleanliness", 0, 5, 3)


    # Encode categorical variables as binary
    # gender_encoded = 1 if gender == "Female" else 0
    customer_type_encoded = 1 if customer_type == "Loyal" else 0
    travel_type_encoded = 1 if travel_type == "Business" else 0
    class_encoded = {"Economy": 0, "Business": 1, "First Class": 2}[flight_class]


    if st.button("Predict Satisfaction"):
        with mlflow.start_run():
                
            expected_features = satisfaction_model.feature_names_in_

        
        # Create input data with only the required features
            input_data = pd.DataFrame({
                
                'Online boarding': [online_boarding],
                'Customer Type': [customer_type_encoded],
                'Type of Travel': [travel_type_encoded],
                'Class': [class_encoded],
                'Inflight wifi service': [inflight_wifi],
                'Inflight entertainment': [inflight_entertainment],
                'On-board service': [on_board_service],
                'Leg room service': [leg_room_service],
                'Seat comfort': [seat_comfort],
                'Checkin service': [checkin_service]

                # 'Departure/Arrival time convenient': [departure_arrival_convenient],
                # 'Ease of Online booking': [ease_booking],
                # 'Gate location': [gate_location],
                # 'Food and drink': [food_drink],         
                # 'Baggage handling': [baggage_handling],
                # 'Inflight service': [inflight_service],
                # 'Cleanliness': [cleanliness],
                # 'Departure Delay in Minutes': [departure_delay],
                # 'Arrival Delay in Minutes': [arrival_delay]
            })

                
            # input_data['id'] = 0  # Use a default value like 0 or any placeholder

            # expected_columns = satisfaction_model.get_booster().feature_names
            # expected_columns = [col for col in expected_columns if col != 'id']

            for col in expected_features:
                if col not in input_data.columns:
                    input_data[col] = 0  # Default value for missing features

        # Reorder columns to match the model's expected input
            input_data = input_data[expected_features]


            input_data = input_data.astype(float)


            # input_data = input_data.astype(float)  # Ensure all values are numeric
            satisfaction_pred = satisfaction_model.predict(input_data)[0]

            # satisfaction_pred_proba = satisfaction_model.predict_proba(input_data)
            # st.write("Prediction Probabilities:", satisfaction_pred_proba)

            mlflow.log_params(input_data.to_dict(orient='records')[0])
            mlflow.log_metric("Predicted Satisfaction", satisfaction_pred)

            st.success(f"Predicted Customer Satisfaction: {'Satisfied' if satisfaction_pred == 1 else 'Not Satisfied'}")


elif choice == "Data Visualization" :

    st.title("Data Visualization")
    vis_category = st.selectbox("Select Category", ["Flight Price Analysis", "Customer Satisfaction Analysis"])
    
    if vis_category == "Flight Price Analysis":
        option = st.selectbox("Select Visualization", [
            "Distribution of Flight Prices", "Flight Count Based on Number of Stops",
            "Flight Duration vs. Price", "Impact of Total Stops on Flight Price",
            "Flight Price Variation Across Airlines", "Airline vs Route vs Price",
            "Total Stops vs Duration vs Price"])
        
        if option == "Distribution of Flight Prices":
            plt.figure(figsize=(10,5))
            sns.histplot(flight_eda['Price'])
            plt.xlabel('Flight prices')
            plt.ylabel('Frequency')
            plt.title("Distribution of Flight Prices")
            fig = plt.gcf()
            st.pyplot(fig)
        elif option == "Flight Count Based on Number of Stops":
            plt.figure(figsize=(8, 5))
            sns.countplot(data=flight_eda, x='Total_Stops', hue='Total_Stops', palette="magma")
            plt.title("Flight Count Based on Number of Stops")
            plt.xlabel("Number of Stops")
            plt.ylabel("Count")
            fig = plt.gcf()
            st.pyplot(fig)
        elif option == "Flight Duration vs. Price":
            plt.figure(figsize=(10,6))
            sns.scatterplot(x=flight_eda['Duration_Hours'], y=flight_data['Price'], alpha=0.6)
            plt.xlabel('Duration in hours')
            plt.ylabel('Price')
            plt.title("Flight Duration vs. Price")
            fig = plt.gcf()
            st.pyplot(fig)
        elif option == "Impact of Total Stops on Flight Price":
            plt.figure(figsize=(10,6))
            sns.lineplot(data=flight_eda, x='Total_Stops', y='Price')
            plt.xlabel('No. of intermediate stops')
            plt.ylabel('Price')
            plt.title("Impact of Total Stops on Flight Price")
            fig = plt.gcf()
            st.pyplot(fig)
        elif option == "Flight Price Variation Across Airlines":
            plt.figure(figsize=(10,6))
            sns.boxplot(data=flight_eda, x='Airline', y='Price', hue='Airline', palette='viridis')
            plt.xlabel('Airline')
            plt.xticks(rotation=60)
            plt.ylabel('Price')
            plt.title("Flight Price Variation Across Airlines")
            fig = plt.gcf()
            st.pyplot(fig)
        elif option == "Airline vs Route vs Price":
            plt.figure(figsize=(15, 6))
            top_routes = flight_eda['Route'].value_counts().head(10).index
            sns.scatterplot(x='Airline', y='Price', hue='Route', data=flight_eda[flight_eda['Route'].isin(top_routes)])
            plt.xticks(rotation=45)
            plt.title("Airline vs Route vs Price")
            fig = plt.gcf()
            st.pyplot(fig)
        elif option == "Total Stops vs Duration vs Price":
            plt.figure(figsize=(12, 6))
            sns.scatterplot(x=flight_eda['Duration_Hours'], y=flight_eda['Price'], hue=flight_eda['Total_Stops'])
            plt.title("Total Stops vs Duration vs Price")
            plt.xlabel("Duration (in hours)")
            plt.ylabel("Price")
            plt.legend(title="Total Stops")
            fig = plt.gcf()
            st.pyplot(fig)
    
    elif vis_category == "Customer Satisfaction Analysis":
        option = st.selectbox("Select Visualization", ["Categorical Feature Distribution", "Customer Satisfaction Rates", "Flight Distance vs. Satisfaction"])
        
        if option == "Categorical Feature Distribution":
            cat_features = ["Gender", "Customer Type", "Type of Travel", "Class"]
            for col in cat_features:
                plt.figure(figsize=(6, 4))
                ax = sns.countplot(x=customer_eda[col], hue=customer_eda['satisfaction'], palette="viridis")
                sns.move_legend(ax, loc = 'lower left', bbox_to_anchor = (1, 0.5))
                plt.title(f"Distribution of {col}")
                fig = plt.gcf()
                st.pyplot(fig)
        elif option == "Customer Satisfaction Rates":
            sns.histplot(customer_eda['satisfaction'], color='brown')
            plt.xlabel('Customer Satisfaction')
            plt.title('Customer Satisfaction Rates')
            fig = plt.gcf()
            st.pyplot(fig)
        elif option == "Flight Distance vs. Satisfaction":
            sns.boxplot(x="satisfaction", y="Flight Distance", data=customer_eda, hue='satisfaction', palette="coolwarm")
            plt.title("Flight Distance vs. Satisfaction")
            fig = plt.gcf()
            st.pyplot(fig)
