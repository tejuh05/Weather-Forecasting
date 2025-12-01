import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class MangaloreWeatherForecastingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mangalore Weather Forecasting - Karnataka, India")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Mangalore climate parameters
        self.location = "Mangalore, Karnataka, India"
        self.latitude = 12.9141
        self.longitude = 74.8560
        
        # Initialize data
        self.generate_mangalore_weather_data()
        self.preprocess_data()
        
        # Create UI
        self.create_widgets()
        
    def generate_mangalore_weather_data(self):
        """Generate 3 years of realistic weather data for Mangalore, Karnataka"""
        print(f"Generating weather data for {self.location}...")
        
        # Date range for 3 years
        start_date = datetime.now() - timedelta(days=3*365)
        dates = pd.date_range(start=start_date, periods=3*365, freq='D')
        
        np.random.seed(42)  # For reproducible results
        
        # Mangalore climate characteristics:
        # - Tropical monsoon climate
        # - Temperature: 22°C to 35°C year-round
        # - Heavy monsoon: June-September
        # - Pre-monsoon: March-May (hottest)
        # - Post-monsoon: October-February (cooler)
        
        day_of_year = dates.dayofyear
        
        # Generate base temperature with Mangalore's seasonal pattern
        # Hottest: April-May (32-35°C), Coolest: December-January (22-28°C)
        base_temp = 28 + 4 * np.sin(2 * np.pi * (day_of_year - 105) / 365)  # Peak in April
        
        # Add daily temperature variation and slight random variation
        daily_variation = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 1)  # Daily cycle
        noise = np.random.normal(0, 1.5, len(dates))  # Small random variation
        
        temperature = base_temp + daily_variation + noise
        # Constrain to Mangalore's realistic temperature range
        temperature = np.clip(temperature, 20, 37)
        
        # Generate humidity (Mangalore is very humid: 70-90%)
        # Higher during monsoon, slightly lower in dry season
        base_humidity = 80 + 8 * np.sin(2 * np.pi * (day_of_year - 150) / 365)  # Peak during monsoon
        humidity_noise = np.random.normal(0, 5, len(dates))
        humidity = base_humidity + humidity_noise
        humidity = np.clip(humidity, 65, 95)
        
        # Generate wind speed (coastal city, moderate winds)
        # Higher during monsoon season
        monsoon_factor = 1 + 0.5 * np.maximum(0, np.sin(2 * np.pi * (day_of_year - 150) / 365))
        wind_speed = np.random.exponential(8, len(dates)) * monsoon_factor
        wind_speed = np.clip(wind_speed, 3, 25)  # Coastal winds
        
        # Generate rainfall (Heavy monsoon pattern)
        # Mangalore receives 3000+ mm annually, mostly June-September
        rainfall = np.zeros(len(dates))
        
        for i, day in enumerate(day_of_year):
            if 152 <= day <= 275:  # June 1 to October 2 (monsoon)
                # Heavy monsoon rainfall
                if np.random.random() < 0.7:  # 70% chance of rain during monsoon
                    rainfall[i] = np.random.exponential(15) + np.random.gamma(2, 5)
                else:
                    rainfall[i] = np.random.exponential(2)
            elif 60 <= day <= 151:  # March to May (pre-monsoon)
                # Occasional pre-monsoon showers
                if np.random.random() < 0.3:  # 30% chance
                    rainfall[i] = np.random.exponential(8)
            else:  # Post-monsoon and winter
                # Light rainfall
                if np.random.random() < 0.2:  # 20% chance
                    rainfall[i] = np.random.exponential(3)
        
        rainfall = np.clip(rainfall, 0, 150)  # Max daily rainfall
        
        # Adjust temperature based on rainfall (cooler on rainy days)
        temperature = temperature - 0.1 * rainfall
        temperature = np.clip(temperature, 20, 37)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'rainfall': rainfall
        })
        
        print(f"Generated {len(self.data)} days of Mangalore weather data")
        print(f"Temperature range: {self.data['temperature'].min():.1f}°C to {self.data['temperature'].max():.1f}°C")
        print(f"Average humidity: {self.data['humidity'].mean():.1f}%")
        print(f"Total annual rainfall: {self.data['rainfall'].sum()/3:.0f}mm (3-year average)")
        
    def preprocess_data(self):
        """Clean and preprocess the weather data for Mangalore"""
        print("Preprocessing Mangalore weather data...")
        
        # Handle any missing values (simulate some)
        missing_indices = np.random.choice(len(self.data), size=int(0.005 * len(self.data)), replace=False)
        self.data.loc[missing_indices, 'temperature'] = np.nan
        
        # Fill missing values with interpolation
        self.data['temperature'] = self.data['temperature'].interpolate(method='linear')
        
        # Create additional features
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
        self.data['month'] = self.data['date'].dt.month
        
        # Define Mangalore seasons
        def get_mangalore_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Pre-Monsoon'
            elif month in [6, 7, 8, 9]:
                return 'Monsoon'
            else:  # 10, 11
                return 'Post-Monsoon'
                
        self.data['season'] = self.data['month'].apply(get_mangalore_season)
        
        # Create seasonal features (sine/cosine for cyclical nature)
        self.data['sin_day'] = np.sin(2 * np.pi * self.data['day_of_year'] / 365)
        self.data['cos_day'] = np.cos(2 * np.pi * self.data['day_of_year'] / 365)
        
        # Monsoon indicator (important for Mangalore)
        self.data['is_monsoon'] = ((self.data['month'] >= 6) & (self.data['month'] <= 9)).astype(int)
        
        # Create lagged features (3 days)
        for lag in range(1, 4):
            self.data[f'temp_lag_{lag}'] = self.data['temperature'].shift(lag)
            
        # Create moving averages
        self.data['temp_ma_3'] = self.data['temperature'].rolling(window=3).mean()
        self.data['temp_ma_7'] = self.data['temperature'].rolling(window=7).mean()
        
        # Create rainfall indicators
        self.data['rain_yesterday'] = (self.data['rainfall'].shift(1) > 5).astype(int)
        self.data['heavy_rain_3days'] = (self.data['rainfall'].rolling(3).sum() > 50).astype(int)
        
        # Drop rows with NaN values from lagged features
        self.data = self.data.dropna().reset_index(drop=True)
        
        print(f"Preprocessed data shape: {self.data.shape}")
        
    def create_prediction_model(self):
        """Create and train the temperature prediction model for Mangalore"""
        print("Training Mangalore-specific prediction model...")
        
        # Feature columns optimized for Mangalore climate
        feature_cols = ['humidity', 'wind_speed', 'rainfall', 'sin_day', 'cos_day', 'is_monsoon'] + \
                      [f'temp_lag_{i}' for i in range(1, 4)] + \
                      ['temp_ma_3', 'temp_ma_7', 'rain_yesterday', 'heavy_rain_3days']
        
        X = self.data[feature_cols]
        y = self.data['temperature']
        
        # Split data (use last 60 days for validation)
        split_idx = len(X) - 60
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print(f"Model trained - Training MAE: {train_mae:.2f}°C, Validation MAE: {val_mae:.2f}°C")
        
        return feature_cols
        
    def predict_next_week_mangalore(self):
        """Predict temperature for the next 7 days with Mangalore-specific patterns"""
        feature_cols = self.create_prediction_model()
        
        # Get recent data for baseline
        recent_data = self.data.tail(7).copy()
        last_temp = recent_data['temperature'].iloc[-1]
        
        predictions = []
        prediction_dates = []
        
        # Use actual weather patterns for predictions
        last_date = self.data['date'].iloc[-1]
        
        for day in range(1, 8):
            # Create date for prediction
            pred_date = last_date + timedelta(days=day)
            prediction_dates.append(pred_date)
            
            # Calculate seasonal component
            day_of_year = pred_date.timetuple().tm_yday
            sin_day = np.sin(2 * np.pi * day_of_year / 365)
            cos_day = np.cos(2 * np.pi * day_of_year / 365)
            
            # Check if it's monsoon season
            is_monsoon = 1 if 6 <= pred_date.month <= 9 else 0
            
            # Use recent averages for weather variables (Mangalore patterns)
            if is_monsoon:
                # Monsoon season patterns
                avg_humidity = min(85 + np.random.normal(0, 3), 95)
                avg_wind = 12 + np.random.normal(0, 2)
                avg_rainfall = np.random.exponential(10) if np.random.random() < 0.7 else 0
            else:
                # Non-monsoon patterns
                avg_humidity = min(75 + np.random.normal(0, 3), 90)
                avg_wind = 8 + np.random.normal(0, 2)
                avg_rainfall = np.random.exponential(2) if np.random.random() < 0.2 else 0
            
            # Get recent temperature data for lagged features
            if day == 1:
                temp_lag_1 = recent_data['temperature'].iloc[-1]
                temp_lag_2 = recent_data['temperature'].iloc[-2]
                temp_lag_3 = recent_data['temperature'].iloc[-3]
            elif day == 2:
                temp_lag_1 = predictions[0] if predictions else last_temp
                temp_lag_2 = recent_data['temperature'].iloc[-1]
                temp_lag_3 = recent_data['temperature'].iloc[-2]
            elif day == 3:
                temp_lag_1 = predictions[1] if len(predictions) > 1 else last_temp
                temp_lag_2 = predictions[0] if predictions else last_temp
                temp_lag_3 = recent_data['temperature'].iloc[-1]
            else:
                temp_lag_1 = predictions[day-2]
                temp_lag_2 = predictions[day-3]
                temp_lag_3 = predictions[day-4]
            
            # Calculate moving averages
            if len(predictions) >= 2:
                temp_ma_3 = np.mean([predictions[-1], predictions[-2], temp_lag_1])
            else:
                temp_ma_3 = recent_data['temp_ma_3'].iloc[-1]
                
            temp_ma_7 = recent_data['temp_ma_7'].iloc[-1]
            
            # Rainfall indicators
            rain_yesterday = 1 if avg_rainfall > 5 else 0
            heavy_rain_3days = 1 if np.random.random() < 0.3 and is_monsoon else 0
            
            # Prepare features
            features = np.array([[
                avg_humidity, avg_wind, avg_rainfall, sin_day, cos_day, is_monsoon,
                temp_lag_1, temp_lag_2, temp_lag_3, temp_ma_3, temp_ma_7,
                rain_yesterday, heavy_rain_3days
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            pred_temp = self.model.predict(features_scaled)[0]
            
            # Apply Mangalore-specific constraints
            # 1. Don't allow extreme jumps (max 3°C change per day in Mangalore)
            if predictions:
                max_change = 3.0
                if pred_temp - predictions[-1] > max_change:
                    pred_temp = predictions[-1] + max_change
                elif pred_temp - predictions[-1] < -max_change:
                    pred_temp = predictions[-1] - max_change
            else:
                max_change = 3.0
                if pred_temp - last_temp > max_change:
                    pred_temp = last_temp + max_change
                elif pred_temp - last_temp < -max_change:
                    pred_temp = last_temp - max_change
            
            # 2. Apply Mangalore temperature bounds (20-37°C)
            pred_temp = np.clip(pred_temp, 20, 37)
            
            # 3. Apply seasonal constraints for Mangalore
            if is_monsoon:
                # Monsoon season: cooler and more stable (24-32°C)
                pred_temp = np.clip(pred_temp, 24, 32)
            elif pred_date.month in [3, 4, 5]:
                # Pre-monsoon: hottest period (26-35°C)
                pred_temp = np.clip(pred_temp, 26, 35)
            else:
                # Post-monsoon/Winter: mild (22-30°C)
                pred_temp = np.clip(pred_temp, 22, 30)
            
            # 4. Rainfall effect (cooler on rainy days)
            if avg_rainfall > 10:
                pred_temp = pred_temp - min(2, avg_rainfall * 0.05)
                pred_temp = max(pred_temp, 22)  # Don't go too low
            
            predictions.append(pred_temp)
            
        print(f"7-day Mangalore predictions: {[f'{p:.1f}°C' for p in predictions]}")
        return prediction_dates, predictions
        
    def create_widgets(self):
        """Create the Tkinter UI"""
        # Main title
        title_label = tk.Label(self.root, text=f"Weather Forecasting - {self.location}", 
                              font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Location info
        location_label = tk.Label(self.root, text=f"Latitude: {self.latitude}°N, Longitude: {self.longitude}°E | Tropical Monsoon Climate", 
                                 font=("Arial", 10), bg='#f0f0f0', fg='#34495e')
        location_label.pack(pady=(0, 5))
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Control panel
        control_frame = tk.LabelFrame(main_frame, text="Controls", font=("Arial", 12, "bold"),
                                     bg='#f0f0f0', fg='#2c3e50')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Date range selection
        date_frame = tk.Frame(control_frame, bg='#f0f0f0')
        date_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(date_frame, text="Date Range:", font=("Arial", 10, "bold"), 
                bg='#f0f0f0').pack(side=tk.LEFT)
        
        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()
        
        # Set default date range (last 90 days)
        end_date = self.data['date'].max()
        start_date = end_date - timedelta(days=90)
        self.start_date_var.set(start_date.strftime('%Y-%m-%d'))
        self.end_date_var.set(end_date.strftime('%Y-%m-%d'))
        
        tk.Label(date_frame, text="From:", bg='#f0f0f0').pack(side=tk.LEFT, padx=(10, 5))
        start_entry = tk.Entry(date_frame, textvariable=self.start_date_var, width=12)
        start_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(date_frame, text="To:", bg='#f0f0f0').pack(side=tk.LEFT, padx=(0, 5))
        end_entry = tk.Entry(date_frame, textvariable=self.end_date_var, width=12)
        end_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Buttons
        button_frame = tk.Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Button(button_frame, text="View Historical Data", command=self.plot_historical,
                 bg='#3498db', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(button_frame, text="Generate Mangalore Forecast", command=self.plot_forecast,
                 bg='#e74c3c', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(button_frame, text="Show Climate Statistics", command=self.show_statistics,
                 bg='#27ae60', fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(f"Ready - {self.location} Weather Forecasting System")
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                             anchor=tk.W, bg='#ecf0f1', fg='#2c3e50')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def plot_historical(self):
        """Plot historical temperature data for Mangalore"""
        try:
            start_date = datetime.strptime(self.start_date_var.get(), '%Y-%m-%d')
            end_date = datetime.strptime(self.end_date_var.get(), '%Y-%m-%d')
            
            # Filter data
            mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
            filtered_data = self.data[mask]
            
            if filtered_data.empty:
                messagebox.showwarning("Warning", "No data found for the selected date range!")
                return
                
            # Clear previous plots
            self.fig.clear()
            
            # Create subplots
            ax1 = self.fig.add_subplot(221)
            ax2 = self.fig.add_subplot(222)
            ax3 = self.fig.add_subplot(223)
            ax4 = self.fig.add_subplot(224)
            
            # Temperature and rainfall trend
            ax1_twin = ax1.twinx()
            line1 = ax1.plot(filtered_data['date'], filtered_data['temperature'], 
                    color='#e74c3c', linewidth=2, alpha=0.8, label='Temperature')
            bars = ax1_twin.bar(filtered_data['date'], filtered_data['rainfall'], 
                               alpha=0.3, color='#3498db', label='Rainfall', width=0.8)
            
            ax1.set_title('Temperature & Rainfall (Mangalore)', fontweight='bold')
            ax1.set_ylabel('Temperature (°C)', color='#e74c3c')
            ax1_twin.set_ylabel('Rainfall (mm)', color='#3498db')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Seasonal temperature patterns
            seasonal_temps = filtered_data.groupby('season')['temperature'].agg(['mean', 'std'])
            seasons = seasonal_temps.index
            ax2.bar(seasons, seasonal_temps['mean'], yerr=seasonal_temps['std'], 
                   color=['#f39c12', '#27ae60', '#3498db', '#e67e22'], alpha=0.7, capsize=5)
            ax2.set_title('Temperature by Season (Mangalore)', fontweight='bold')
            ax2.set_ylabel('Temperature (°C)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Monthly rainfall pattern (important for Mangalore)
            monthly_rain = filtered_data.groupby('month')['rainfall'].sum()
            ax3.bar(monthly_rain.index, monthly_rain.values, color='#3498db', alpha=0.7)
            ax3.set_title('Monthly Rainfall Pattern', fontweight='bold')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Total Rainfall (mm)')
            ax3.grid(True, alpha=0.3)
            
            # Humidity vs Temperature (important for coastal Mangalore)
            colors = filtered_data['month'].map({12:'blue', 1:'blue', 2:'blue',
                                               3:'orange', 4:'orange', 5:'orange',
                                               6:'green', 7:'green', 8:'green', 9:'green',
                                               10:'red', 11:'red'})
            ax4.scatter(filtered_data['humidity'], filtered_data['temperature'], 
                       alpha=0.6, c=colors, s=15)
            ax4.set_title('Humidity vs Temperature\n(Blue:Winter, Orange:Pre-monsoon, Green:Monsoon, Red:Post-monsoon)', 
                         fontweight='bold', fontsize=10)
            ax4.set_xlabel('Humidity (%)')
            ax4.set_ylabel('Temperature (°C)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.canvas.draw()
            
            self.status_var.set(f"Displayed Mangalore weather data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting historical data: {str(e)}")
            
    def plot_forecast(self):
        """Plot 7-day temperature forecast for Mangalore"""
        try:
            self.status_var.set("Generating Mangalore forecast... Please wait.")
            self.root.update()
            
            # Generate predictions
            pred_dates, predictions = self.predict_next_week_mangalore()
            
            # Clear previous plots
            self.fig.clear()
            
            # Create subplot
            ax = self.fig.add_subplot(111)
            
            # Plot last 30 days of historical data
            recent_data = self.data.tail(30)
            ax.plot(recent_data['date'], recent_data['temperature'], 
                   color='#2c3e50', linewidth=2, label='Historical', marker='o', markersize=3)
            
            # Plot forecast
            ax.plot(pred_dates, predictions, color='#e74c3c', linewidth=3, 
                   label='7-Day Mangalore Forecast', marker='s', markersize=6, linestyle='--')
            
            # Add realistic confidence interval for Mangalore
            forecast_std = 1.5  # ±1.5°C confidence interval (realistic for Mangalore)
            upper_bound = np.array(predictions) + forecast_std
            lower_bound = np.array(predictions) - forecast_std
            
            ax.fill_between(pred_dates, lower_bound, upper_bound, 
                           alpha=0.3, color='#e74c3c', label='Confidence Interval (±1.5°C)')
            
            # Check if prediction period includes monsoon
            current_month = pred_dates[0].month
            season_info = ""
            if 6 <= current_month <= 9:
                season_info = " - Monsoon Season"
            elif current_month in [3, 4, 5]:
                season_info = " - Pre-Monsoon (Hot & Humid)"
            elif current_month in [10, 11]:
                season_info = " - Post-Monsoon"
            else:
                season_info = " - Winter Season"
                
            ax.set_title(f'Mangalore Temperature Forecast - Next 7 Days{season_info}', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Set realistic y-axis limits for Mangalore
            ax.set_ylim(18, 40)
            
            # Add forecast values as text
            for i, (date, temp) in enumerate(zip(pred_dates, predictions)):
                ax.annotate(f'{temp:.1f}°C', (date, temp), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=9, fontweight='bold', color='#e74c3c')
            
            plt.tight_layout()
            self.canvas.draw()
            
            # Show forecast summary
            avg_temp = np.mean(predictions)
            min_temp = np.min(predictions)
            max_temp = np.max(predictions)
            
            # Determine comfort level for Mangalore
            if avg_temp <= 28:
                comfort = "Pleasant"
            elif avg_temp <= 32:
                comfort = "Warm & Humid"
            else:
                comfort = "Hot & Very Humid"
            
            forecast_text = f"""Mangalore 7-Day Forecast Summary:

Average Temperature: {avg_temp:.1f}°C
Minimum: {min_temp:.1f}°C  
Maximum: {max_temp:.1f}°C
Comfort Level: {comfort}

Season: {season_info.strip(' -')}
Expected Conditions:
• High humidity (75-90%)
• Coastal breeze
• {'Heavy rainfall likely' if 6 <= current_month <= 9 else 'Light/No rainfall'}

Daily Predictions: {', '.join([f'{p:.1f}°C' for p in predictions])}

Note: Predictions based on Mangalore's tropical monsoon climate patterns"""
            
            messagebox.showinfo("Mangalore Forecast Summary", forecast_text)
            
            self.status_var.set("Mangalore 7-day forecast generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating forecast: {str(e)}")
            self.status_var.set("Error generating forecast")
            
    def show_statistics(self):
        """Show comprehensive weather statistics for Mangalore"""
        try:
            start_date = datetime.strptime(self.start_date_var.get(), '%Y-%m-%d')
            end_date = datetime.strptime(self.end_date_var.get(), '%Y-%m-%d')
            
            # Filter data
            mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
            filtered_data = self.data[mask]
            
            if filtered_data.empty:
                messagebox.showwarning("Warning", "No data found for the selected date range!")
                return
            
            # Calculate statistics
            temp_stats = filtered_data['temperature'].describe()
            humidity_stats = filtered_data['humidity'].describe()
            wind_stats = filtered_data['wind_speed'].describe()
            rainfall_stats = filtered_data['rainfall'].describe()
            
            # Seasonal statistics
            seasonal_stats = filtered_data.groupby('season')['temperature'].agg(['mean', 'min', 'max', 'std'])
            # Ensure all seasons are present to avoid KeyError
            seasons_list = ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']
            for season in seasons_list:
                if season not in seasonal_stats.index:
                    seasonal_stats.loc[season] = [float('nan')] * 4
            seasonal_stats = seasonal_stats.sort_index()
            
            # Monsoon analysis
            monsoon_data = filtered_data[filtered_data['is_monsoon'] == 1]
            non_monsoon_data = filtered_data[filtered_data['is_monsoon'] == 0]
            
            # Rainfall analysis
            rainy_days = len(filtered_data[filtered_data['rainfall'] > 1])
            heavy_rain_days = len(filtered_data[filtered_data['rainfall'] > 25])
            total_rainfall = filtered_data['rainfall'].sum()
            
            # Format statistics
            stats_text = f"""MANGALORE CLIMATE STATISTICS
{self.location}
Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}

═══════════════════════════════════════

TEMPERATURE ANALYSIS (°C):
Overall Statistics:
• Mean: {temp_stats['mean']:.1f}°C
• Standard Deviation: {temp_stats['std']:.1f}°C
• Minimum: {temp_stats['min']:.1f}°C
• Maximum: {temp_stats['max']:.1f}°C
• Median: {temp_stats['50%']:.1f}°C

Seasonal Breakdown:
• Winter (Dec-Feb): {seasonal_stats.loc['Winter', 'mean']:.1f}°C (Range: {seasonal_stats.loc['Winter', 'min']:.1f}°C - {seasonal_stats.loc['Winter', 'max']:.1f}°C)
• Pre-Monsoon (Mar-May): {seasonal_stats.loc['Pre-Monsoon', 'mean']:.1f}°C (Range: {seasonal_stats.loc['Pre-Monsoon', 'min']:.1f}°C - {seasonal_stats.loc['Pre-Monsoon', 'max']:.1f}°C)
• Monsoon (Jun-Sep): {seasonal_stats.loc['Monsoon', 'mean']:.1f}°C (Range: {seasonal_stats.loc['Monsoon', 'min']:.1f}°C - {seasonal_stats.loc['Monsoon', 'max']:.1f}°C)
• Post-Monsoon (Oct-Nov): {seasonal_stats.loc['Post-Monsoon', 'mean']:.1f}°C (Range: {seasonal_stats.loc['Post-Monsoon', 'min']:.1f}°C - {seasonal_stats.loc['Post-Monsoon', 'max']:.1f}°C)

═══════════════════════════════════════

HUMIDITY ANALYSIS (%):
• Mean Humidity: {humidity_stats['mean']:.1f}%
• Standard Deviation: {humidity_stats['std']:.1f}%
• Minimum: {humidity_stats['min']:.1f}%
• Maximum: {humidity_stats['max']:.1f}%
• Typical Range: {humidity_stats['25%']:.1f}% - {humidity_stats['75%']:.1f}%

Monsoon vs Non-Monsoon Humidity:
• Monsoon Season: {monsoon_data['humidity'].mean():.1f}% avg
• Non-Monsoon: {non_monsoon_data['humidity'].mean():.1f}% avg

═══════════════════════════════════════

WIND ANALYSIS (km/h):
• Mean Wind Speed: {wind_stats['mean']:.1f} km/h
• Standard Deviation: {wind_stats['std']:.1f} km/h
• Maximum Gust: {wind_stats['max']:.1f} km/h
• Typical Coastal Breeze: {wind_stats['50%']:.1f} km/h

═══════════════════════════════════════

RAINFALL ANALYSIS (Mangalore Monsoon Pattern):
• Total Rainfall: {total_rainfall:.0f} mm
• Annual Estimate: {(total_rainfall * 365 / len(filtered_data)):.0f} mm/year
• Average Daily: {rainfall_stats['mean']:.1f} mm
• Maximum Single Day: {rainfall_stats['max']:.1f} mm
• Rainy Days (>1mm): {rainy_days} days ({100*rainy_days/len(filtered_data):.1f}% of period)
• Heavy Rain Days (>25mm): {heavy_rain_days} days ({100*heavy_rain_days/len(filtered_data):.1f}% of period)

Monsoon Rainfall Impact:
• Monsoon Season Avg Temp: {monsoon_data['temperature'].mean():.1f}°C
• Non-Monsoon Avg Temp: {non_monsoon_data['temperature'].mean():.1f}°C
• Cooling Effect: {non_monsoon_data['temperature'].mean() - monsoon_data['temperature'].mean():.1f}°C lower during monsoon

═══════════════════════════════════════

COMFORT INDEX:
• Pleasant Days (<28°C): {len(filtered_data[filtered_data['temperature'] < 28])} days ({100*len(filtered_data[filtered_data['temperature'] < 28])/len(filtered_data):.1f}%)
• Warm Days (28-32°C): {len(filtered_data[(filtered_data['temperature'] >= 28) & (filtered_data['temperature'] < 32)])} days ({100*len(filtered_data[(filtered_data['temperature'] >= 28) & (filtered_data['temperature'] < 32)])/len(filtered_data):.1f}%)
• Hot Days (>32°C): {len(filtered_data[filtered_data['temperature'] >= 32])} days ({100*len(filtered_data[filtered_data['temperature'] >= 32])/len(filtered_data):.1f}%)

═══════════════════════════════════════

CLIMATE NOTES:
• Mangalore experiences a tropical monsoon climate
• High humidity year-round due to coastal location
• Significant rainfall during June-September monsoon
• Temperature moderated by Arabian Sea proximity
• Pre-monsoon period (March-May) is hottest & most humid

Total days analyzed: {len(filtered_data)}"""
            
            # Create statistics window
            stats_window = tk.Toplevel(self.root)
            stats_window.title(f"Mangalore Climate Statistics - {self.location}")
            stats_window.geometry("600x700")
            stats_window.configure(bg='#f0f0f0')
            
            # Create scrollable text widget
            frame = tk.Frame(stats_window, bg='#f0f0f0')
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(frame, font=("Courier", 9), 
                                 bg='white', fg='#2c3e50', wrap=tk.WORD)
            scrollbar = tk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            text_widget.insert(tk.END, stats_text)
            text_widget.config(state=tk.DISABLED)
            
            self.status_var.set("Mangalore climate statistics displayed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error showing statistics: {str(e)}")

def main():
    """Main function to run the Mangalore weather application"""
    print("Starting Mangalore Weather Forecasting Application...")
    print("Location: Mangalore, Karnataka, India")
    print("Climate Type: Tropical Monsoon")
    print("Coordinates: 12.9141°N, 74.8560°E")
    
    root = tk.Tk()
    app = MangaloreWeatherForecastingApp(root)
    
    print("\nApplication loaded successfully!")
    print("Mangalore-Specific Features:")
    print("✓ Tropical monsoon climate modeling")
    print("✓ Seasonal temperature patterns (20-37°C range)")
    print("✓ Monsoon rainfall prediction (June-September)")
    print("✓ High humidity modeling (65-95%)")
    print("✓ Coastal wind patterns")
    print("✓ Pre-monsoon hot season effects")
    print("✓ Realistic 7-day forecasts with seasonal constraints")
    
    print("\nTemperature Ranges by Season:")
    print("• Winter (Dec-Feb): 22-28°C")
    print("• Pre-Monsoon (Mar-May): 26-35°C") 
    print("• Monsoon (Jun-Sep): 24-32°C")
    print("• Post-Monsoon (Oct-Nov): 22-30°C")
    
    root.mainloop()

if __name__ == "__main__":
    main()