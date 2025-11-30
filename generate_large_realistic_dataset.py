"""
Generate large-scale realistic synthetic dataset with 1,000 athletes.
All files are internally consistent with cross-file causality and realistic statistical distributions.
"""

import random
import csv
from datetime import datetime, timedelta
import math
import hashlib

# Set seed for reproducibility
random.seed(42)

# Configuration
NUM_ATHLETES = 1000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
DAYS_IN_PERIOD = (END_DATE - START_DATE).days

print("=" * 80)
print("GENERATING LARGE-SCALE REALISTIC ATHLETE DATASET")
print("=" * 80)
print(f"Athletes: {NUM_ATHLETES}")
print(f"Period: {START_DATE.date()} to {END_DATE.date()} ({DAYS_IN_PERIOD} days)")
print("=" * 80)

# Realistic data pools
FIRST_NAMES_MALE = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
    "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
    "Carlos", "Miguel", "Luis", "Juan", "Marco", "Antonio", "Rafael", "Diego", "Gabriel", "Mateo",
    "Chen", "Wei", "Li", "Wang", "Zhang", "Liu", "Yang", "Huang", "Wu", "Zhou",
    "Mohammed", "Ahmed", "Ali", "Omar", "Hassan", "Youssef", "Khalid", "Mustafa", "Tariq", "Karim",
    "Raj", "Arjun", "Amit", "Rohan", "Vikram", "Aditya", "Sanjay", "Ravi", "Kiran", "Nikhil",
    "Hans", "Klaus", "Stefan", "Felix", "Lukas", "Sebastian", "Markus", "Andreas", "Thomas", "Maximilian",
    "Luca", "Matteo", "Giorgio", "Francesco", "Alessandro", "Leonardo", "Lorenzo", "Davide", "Simone", "Andrea",
    "Yuki", "Takeshi", "Kenji", "Hiroshi", "Ryu", "Kaito", "Haruki", "Sota", "Ren", "Daiki",
    "Olivier", "Pierre", "Jean", "Philippe", "Nicolas", "Alexandre", "Antoine", "Julien", "Laurent", "Vincent"
]

FIRST_NAMES_FEMALE = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan", "Jessica", "Sarah", "Karen",
    "Nancy", "Lisa", "Margaret", "Betty", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Donna",
    "Maria", "Carmen", "Sofia", "Isabella", "Lucia", "Valentina", "Camila", "Ana", "Rosa", "Elena",
    "Li", "Mei", "Ying", "Fang", "Xiu", "Juan", "Min", "Yan", "Hong", "Jing",
    "Fatima", "Aisha", "Layla", "Zainab", "Mariam", "Amina", "Nour", "Sara", "Hana", "Yasmin",
    "Priya", "Anjali", "Neha", "Pooja", "Riya", "Kavya", "Divya", "Shreya", "Ananya", "Meera",
    "Anna", "Sophie", "Emma", "Marie", "Laura", "Julia", "Lena", "Nina", "Mia", "Lisa",
    "Giulia", "Francesca", "Chiara", "Valentina", "Elena", "Sara", "Martina", "Alessia", "Silvia", "Federica",
    "Sakura", "Yuki", "Aiko", "Emi", "Hana", "Miyu", "Rio", "Nana", "Yui", "Miku",
    "Claire", "Camille", "Julie", "Léa", "Manon", "Chloé", "Emma", "Inès", "Lucie", "Marine"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Chen", "Wang", "Zhang", "Li", "Liu", "Yang", "Huang", "Zhao", "Wu", "Zhou",
    "Kumar", "Patel", "Singh", "Shah", "Gupta", "Sharma", "Reddy", "Mehta", "Desai", "Agarwal",
    "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz", "Hoffmann",
    "Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Colombo", "Ricci", "Marino", "Greco",
    "Dupont", "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy",
    "Tanaka", "Suzuki", "Yamamoto", "Watanabe", "Ito", "Nakamura", "Kobayashi", "Sato", "Sasaki", "Kato"
]

CITIES_COUNTRIES = [
    ("New York, USA", "USA", 40.7, -74.0), ("Los Angeles, USA", "USA", 34.0, -118.2), 
    ("Chicago, USA", "USA", 41.8, -87.6), ("Houston, USA", "USA", 29.7, -95.3),
    ("Phoenix, USA", "USA", 33.4, -112.0), ("Philadelphia, USA", "USA", 39.9, -75.1),
    ("San Antonio, USA", "USA", 29.4, -98.4), ("San Diego, USA", "USA", 32.7, -117.1),
    ("Dallas, USA", "USA", 32.7, -96.7), ("San Jose, USA", "USA", 37.3, -121.8),
    ("Austin, USA", "USA", 30.2, -97.7), ("Jacksonville, USA", "USA", 30.3, -81.6),
    ("London, UK", "UK", 51.5, -0.1), ("Manchester, UK", "UK", 53.4, -2.2),
    ("Birmingham, UK", "UK", 52.4, -1.9), ("Leeds, UK", "UK", 53.8, -1.5),
    ("Glasgow, UK", "UK", 55.8, -4.2), ("Liverpool, UK", "UK", 53.4, -2.9),
    ("Berlin, Germany", "Germany", 52.5, 13.4), ("Munich, Germany", "Germany", 48.1, 11.5),
    ("Hamburg, Germany", "Germany", 53.5, 10.0), ("Frankfurt, Germany", "Germany", 50.1, 8.6),
    ("Cologne, Germany", "Germany", 50.9, 6.9), ("Stuttgart, Germany", "Germany", 48.7, 9.1),
    ("Paris, France", "France", 48.8, 2.3), ("Lyon, France", "France", 45.7, 4.8),
    ("Marseille, France", "France", 43.2, 5.3), ("Toulouse, France", "France", 43.6, 1.4),
    ("Madrid, Spain", "Spain", 40.4, -3.7), ("Barcelona, Spain", "Spain", 41.3, 2.1),
    ("Valencia, Spain", "Spain", 39.4, -0.3), ("Seville, Spain", "Spain", 37.3, -5.9),
    ("Rome, Italy", "Italy", 41.9, 12.4), ("Milan, Italy", "Italy", 45.4, 9.1),
    ("Naples, Italy", "Italy", 40.8, 14.2), ("Turin, Italy", "Italy", 45.0, 7.6),
    ("Amsterdam, Netherlands", "Netherlands", 52.3, 4.8), ("Rotterdam, Netherlands", "Netherlands", 51.9, 4.4),
    ("Stockholm, Sweden", "Sweden", 59.3, 18.0), ("Gothenburg, Sweden", "Sweden", 57.7, 11.9),
    ("Copenhagen, Denmark", "Denmark", 55.6, 12.5), ("Oslo, Norway", "Norway", 59.9, 10.7),
    ("Helsinki, Finland", "Finland", 60.1, 24.9), ("Warsaw, Poland", "Poland", 52.2, 21.0),
    ("Tokyo, Japan", "Japan", 35.6, 139.6), ("Osaka, Japan", "Japan", 34.6, 135.5),
    ("Kyoto, Japan", "Japan", 35.0, 135.7), ("Seoul, South Korea", "South Korea", 37.5, 126.9),
    ("Busan, South Korea", "South Korea", 35.1, 129.0), ("Beijing, China", "China", 39.9, 116.4),
    ("Shanghai, China", "China", 31.2, 121.4), ("Guangzhou, China", "China", 23.1, 113.2),
    ("Shenzhen, China", "China", 22.5, 114.1), ("Mumbai, India", "India", 19.0, 72.8),
    ("Delhi, India", "India", 28.6, 77.2), ("Bangalore, India", "India", 12.9, 77.5),
    ("Hyderabad, India", "India", 17.3, 78.4), ("Chennai, India", "India", 13.0, 80.2),
    ("Pune, India", "India", 18.5, 73.8), ("Sydney, Australia", "Australia", -33.8, 151.2),
    ("Melbourne, Australia", "Australia", -37.8, 144.9), ("Brisbane, Australia", "Australia", -27.4, 153.0),
    ("Toronto, Canada", "Canada", 43.6, -79.3), ("Vancouver, Canada", "Canada", 49.2, -123.1),
    ("Montreal, Canada", "Canada", 45.5, -73.5), ("São Paulo, Brazil", "Brazil", -23.5, -46.6),
    ("Rio de Janeiro, Brazil", "Brazil", -22.9, -43.2), ("Brasília, Brazil", "Brazil", -15.7, -47.8),
    ("Mexico City, Mexico", "Mexico", 19.4, -99.1), ("Buenos Aires, Argentina", "Argentina", -34.6, -58.3),
    ("Santiago, Chile", "Chile", -33.4, -70.6), ("Moscow, Russia", "Russia", 55.7, 37.6),
    ("St Petersburg, Russia", "Russia", 59.9, 30.3), ("Dubai, UAE", "UAE", 25.2, 55.2),
    ("Singapore, Singapore", "Singapore", 1.3, 103.8), ("Bangkok, Thailand", "Thailand", 13.7, 100.5),
    ("Istanbul, Turkey", "Turkey", 41.0, 28.9), ("Cairo, Egypt", "Egypt", 30.0, 31.2),
    ("Lagos, Nigeria", "Nigeria", 6.5, 3.3), ("Johannesburg, South Africa", "South Africa", -26.2, 28.0)
]

SPORTS = ["Running", "Cycling", "Swimming", "Triathlon", "Football", "Basketball", "Tennis", 
          "Rowing", "CrossFit", "Hiking", "Climbing", "Skiing", "Volleyball", "Rugby", "Boxing"]

COMPETITION_LEVELS = ["Beginner", "Intermediate", "Advanced", "Elite Amateur", "Semi-Pro"]
COMM_PREFS = ["Text-focused", "Voice-focused", "Mixed", "App-based", "Email-focused"]

INJURY_TYPES = [
    "Muscle strain", "Knee pain", "Ankle sprain", "Lower back pain", "Shin splints", 
    "Tendonitis", "Stress fracture", "IT band syndrome", "Plantar fasciitis", 
    "Shoulder pain", "Hip flexor strain", "Achilles tendonitis", "Hamstring pull",
    "Quadriceps strain", "Rotator cuff injury", "Tennis elbow", "Runner's knee",
    "Meniscus tear", "ACL strain", "MCL sprain", "Calf strain", "Groin pull"
]

EVENT_TYPES = [
    "Competition", "Training milestone", "Location change", "Career change",
    "Injury recovery", "Personal achievement", "Education", "Family event",
    "Race victory", "Personal best", "Training camp", "Coach change",
    "Team change", "Sponsorship deal", "Major competition", "Retirement consideration"
]

def hash_id_to_float(athlete_id, salt=""):
    """Generate deterministic but unique float from athlete_id"""
    hash_obj = hashlib.md5(f"{athlete_id}{salt}".encode())
    return int(hash_obj.hexdigest(), 16) % 10000 / 10000.0

def gaussian_with_variance(mean, std, athlete_id, salt=""):
    """Gaussian distribution with person-specific variance"""
    base_variance = hash_id_to_float(athlete_id, salt) - 0.5
    return random.gauss(mean + base_variance * std * 0.3, std)

def generate_master_profiles(num_athletes):
    """Generate athlete_master_profiles.csv with 1000 athletes"""
    athletes = []
    
    for i in range(1, num_athletes + 1):
        athlete_id = f"ATH_{i:04d}"
        
        # Gender distribution
        gender = random.choices(["Male", "Female"], weights=[0.52, 0.48])[0]
        
        # Age with realistic distribution (more 25-35 year olds)
        age_pools = [(18, 24), (25, 35), (36, 45), (46, 60)]
        age_weights = [0.15, 0.45, 0.30, 0.10]
        age_range = random.choices(age_pools, weights=age_weights)[0]
        age = random.randint(age_range[0], age_range[1])
        
        # Name
        first_name = random.choice(FIRST_NAMES_MALE if gender == "Male" else FIRST_NAMES_FEMALE)
        last_name = random.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"
        
        # Location
        location, nationality, lat, lon = random.choice(CITIES_COUNTRIES)
        
        # Physical attributes with realistic variance
        if gender == "Male":
            height = int(gaussian_with_variance(178, 8, athlete_id, "height"))
            weight = int(gaussian_with_variance(76, 12, athlete_id, "weight"))
        else:
            height = int(gaussian_with_variance(165, 7, athlete_id, "height"))
            weight = int(gaussian_with_variance(63, 10, athlete_id, "weight"))
        
        height = max(150, min(210, height))
        weight = max(45, min(130, weight))
        
        # Sport and level
        sport = random.choice(SPORTS)
        
        # Competition level correlates with age
        if age < 25:
            comp_level = random.choices(COMPETITION_LEVELS, weights=[0.4, 0.35, 0.20, 0.04, 0.01])[0]
        elif age < 35:
            comp_level = random.choices(COMPETITION_LEVELS, weights=[0.15, 0.35, 0.35, 0.12, 0.03])[0]
        else:
            comp_level = random.choices(COMPETITION_LEVELS, weights=[0.10, 0.30, 0.40, 0.18, 0.02])[0]
        
        # Years training
        level_years = {
            "Beginner": (1, 3), 
            "Intermediate": (3, 8), 
            "Advanced": (6, 15), 
            "Elite Amateur": (10, 20),
            "Semi-Pro": (12, 25)
        }
        min_years, max_years = level_years[comp_level]
        max_possible = max(min_years, min(max_years, age - 16))
        years_training = random.randint(min_years, max_possible)
        
        # Social attributes with personality variance
        personality_hash = hash_id_to_float(athlete_id, "personality")
        
        # Extroversion affects network size
        if personality_hash > 0.7:  # Extroverted
            social_network = int(gaussian_with_variance(80, 30, athlete_id, "network"))
        elif personality_hash > 0.3:  # Ambivert
            social_network = int(gaussian_with_variance(50, 20, athlete_id, "network"))
        else:  # Introverted
            social_network = int(gaussian_with_variance(25, 15, athlete_id, "network"))
        
        social_network = max(5, min(200, social_network))
        
        # Coaching
        coaching = 1 if random.random() < (0.4 + comp_level_to_num(comp_level) * 0.15) else 0
        
        # Training partners
        partners = random.randint(0, min(8, int(social_network / 15)))
        
        # Communication preference correlates with personality
        if personality_hash > 0.6:
            comm_pref = random.choice(["Voice-focused", "Mixed"])
        else:
            comm_pref = random.choice(["Text-focused", "App-based", "Email-focused"])
        
        # Career status
        career_status = "Active" if random.random() < 0.95 else "Retired"
        
        # BMI and VO2max estimates
        bmi = round(weight / ((height/100) ** 2), 1)
        
        # VO2max estimate (fitness indicator)
        base_vo2 = 45 if gender == "Male" else 38
        level_bonus = comp_level_to_num(comp_level) * 5
        age_penalty = (age - 25) * 0.15
        vo2max = max(30, int(base_vo2 + level_bonus - age_penalty + random.gauss(0, 3)))
        
        athletes.append({
            "athlete_id": athlete_id,
            "name": name,
            "age": age,
            "gender": gender,
            "location": location,
            "nationality": nationality,
            "latitude": lat,
            "longitude": lon,
            "primary_sport": sport,
            "secondary_sport": random.choice([s for s in SPORTS if s != sport]) if random.random() < 0.3 else "",
            "competition_level": comp_level,
            "years_training": years_training,
            "height_cm": height,
            "weight_kg": weight,
            "bmi": bmi,
            "estimated_vo2max": vo2max,
            "social_network_size": social_network,
            "coaching_support": coaching,
            "training_partners": partners,
            "career_status": career_status,
            "communication_preference": comm_pref,
            "personality_score": round(personality_hash * 10, 1)  # 0-10 scale
        })
    
    return athletes

def comp_level_to_num(level):
    """Convert competition level to numeric"""
    mapping = {"Beginner": 0, "Intermediate": 1, "Advanced": 2, "Elite Amateur": 3, "Semi-Pro": 4}
    return mapping.get(level, 1)

def generate_daily_health_metrics(athletes):
    """Generate daily_health_metrics.csv"""
    metrics = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        age = athlete["age"]
        gender = athlete["gender"]
        comp_level = athlete["competition_level"]
        vo2max = athlete["estimated_vo2max"]
        
        # Base resting HR (fitness-dependent)
        base_resting_hr = 65 - (vo2max - 40) * 0.4 + (age - 30) * 0.2
        base_resting_hr = max(40, min(80, base_resting_hr))
        
        # Number of days (60-180)
        num_days = random.randint(60, 180)
        
        # Generate dates
        dates = sorted([START_DATE + timedelta(days=random.randint(0, DAYS_IN_PERIOD)) 
                       for _ in range(num_days)])
        
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            
            # Daily variance
            resting_hr = int(gaussian_with_variance(base_resting_hr, 4, athlete_id, date_str))
            resting_hr = max(40, min(95, resting_hr))
            
            # Recovery score (inverse correlation with resting HR)
            recovery_base = 100 - (resting_hr - 50)
            recovery_score = int(gaussian_with_variance(recovery_base, 10, athlete_id, f"rec{date_str}"))
            recovery_score = max(30, min(100, recovery_score))
            
            # Sleep
            sleep_hours = round(gaussian_with_variance(7.2, 0.9, athlete_id, f"sleep{date_str}"), 1)
            sleep_hours = max(4.0, min(11.0, sleep_hours))
            
            # Steps (activity-dependent)
            base_steps = 8000 + comp_level_to_num(comp_level) * 1500
            steps = int(gaussian_with_variance(base_steps, 2500, athlete_id, f"steps{date_str}"))
            steps = max(2000, min(25000, steps))
            
            # HRV (heart rate variability)
            base_hrv = 50 + (vo2max - 40) * 0.8
            hrv = int(gaussian_with_variance(base_hrv, 12, athlete_id, f"hrv{date_str}"))
            hrv = max(20, min(120, hrv))
            
            # Stress (random but realistic)
            stress_level = random.randint(1, 10)
            
            # Mood (correlates with stress inversely)
            mood_base = 7 - (stress_level - 5) * 0.3
            mood = int(gaussian_with_variance(mood_base, 1.5, athlete_id, f"mood{date_str}"))
            mood = max(1, min(10, mood))
            
            # Weight variance
            weight = athlete["weight_kg"] + round(random.gauss(0, 0.6), 1)
            
            # Calories burned
            calories = int(1800 + steps * 0.04 + random.gauss(0, 200))
            
            metrics.append({
                "athlete_id": athlete_id,
                "date": date_str,
                "resting_heart_rate": resting_hr,
                "recovery_score": recovery_score,
                "sleep_hours": sleep_hours,
                "daily_steps": steps,
                "heart_rate_variability": hrv,
                "stress_level": stress_level,
                "mood_score": mood,
                "weight_kg": weight,
                "calories_burned": calories
            })
    
    return metrics

def generate_strava_activities(athletes):
    """Generate strava_activities.csv"""
    activities = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        sport = athlete["primary_sport"]
        secondary_sport = athlete.get("secondary_sport", "")
        comp_level = athlete["competition_level"]
        age = athlete["age"]
        vo2max = athlete["estimated_vo2max"]
        
        # Activity frequency based on level
        level_activities = {
            "Beginner": (80, 150),
            "Intermediate": (150, 250),
            "Advanced": (250, 350),
            "Elite Amateur": (300, 400),
            "Semi-Pro": (350, 500)
        }
        min_act, max_act = level_activities[comp_level]
        num_activities = random.randint(min_act, max_act)
        
        for _ in range(num_activities):
            date = START_DATE + timedelta(days=random.randint(0, DAYS_IN_PERIOD))
            date_str = date.strftime("%Y-%m-%d")
            
            # Activity type (75% primary, 20% secondary, 5% other)
            rand = random.random()
            if rand < 0.75:
                activity_type = sport
            elif rand < 0.95 and secondary_sport:
                activity_type = secondary_sport
            else:
                activity_type = random.choice([s for s in SPORTS if s not in [sport, secondary_sport]])
            
            # Distance based on sport and level
            distance, duration, elevation = get_activity_metrics(
                activity_type, comp_level, age, vo2max, athlete_id, date_str
            )
            
            # Heart rate
            max_hr = 220 - age
            intensity = random.gauss(0.72, 0.08)  # 65-80% of max typically
            avg_hr = int(max_hr * max(0.55, min(0.90, intensity)))
            max_hr_activity = int(max_hr * max(0.75, min(0.98, intensity + 0.15)))
            
            # Calories (based on duration and intensity)
            base_cal_per_min = 8 if activity_type in ["Running", "Cycling", "Rowing"] else 6
            calories = int(duration * base_cal_per_min * intensity * 1.5 + random.gauss(0, 50))
            calories = max(50, calories)
            
            # Pace (for running/cycling)
            if activity_type == "Running" and distance > 0:
                pace_min_km = round(duration / distance, 2)
            elif activity_type == "Cycling" and distance > 0:
                pace_km_h = round(distance / (duration / 60), 2)
            else:
                pace_min_km = 0
                pace_km_h = 0
            
            activities.append({
                "athlete_id": athlete_id,
                "date": date_str,
                "activity_type": activity_type,
                "distance_km": distance,
                "duration_minutes": duration,
                "elevation_gain_m": elevation,
                "avg_heart_rate": avg_hr,
                "max_heart_rate": max_hr_activity,
                "calories_burned": calories,
                "pace_min_per_km": pace_min_km if activity_type == "Running" else 0,
                "avg_speed_km_h": pace_km_h if activity_type == "Cycling" else 0
            })
    
    return activities

def get_activity_metrics(activity_type, comp_level, age, vo2max, athlete_id, date_str):
    """Calculate realistic distance, duration, and elevation"""
    level_num = comp_level_to_num(comp_level)
    
    if activity_type == "Running":
        base_dist = 6 + level_num * 2.5
        distance = round(gaussian_with_variance(base_dist, 3, athlete_id, f"dist{date_str}"), 2)
        distance = max(2, min(42, distance))
        
        # Pace affected by age and fitness
        base_pace = 6.5 - (vo2max - 40) * 0.05 + (age - 30) * 0.02
        pace = gaussian_with_variance(base_pace, 0.7, athlete_id, f"pace{date_str}")
        duration = int(distance * max(4, min(9, pace)))
        
        # Elevation
        elevation = int(distance * random.gauss(12, 8))
        elevation = max(0, min(int(distance * 50), elevation))
        
    elif activity_type == "Cycling":
        base_dist = 30 + level_num * 15
        distance = round(gaussian_with_variance(base_dist, 15, athlete_id, f"dist{date_str}"), 2)
        distance = max(10, min(180, distance))
        
        # Speed affected by fitness
        base_speed = 22 + (vo2max - 40) * 0.15
        speed = gaussian_with_variance(base_speed, 4, athlete_id, f"speed{date_str}")
        duration = int(distance / max(15, min(40, speed)) * 60)
        
        elevation = int(distance * random.gauss(18, 12))
        elevation = max(0, min(int(distance * 60), elevation))
        
    elif activity_type == "Swimming":
        base_dist = 1.5 + level_num * 0.8
        distance = round(gaussian_with_variance(base_dist, 0.6, athlete_id, f"dist{date_str}"), 2)
        distance = max(0.5, min(10, distance))
        duration = int(distance * random.gauss(25, 5))
        elevation = 0
        
    elif activity_type == "Triathlon":
        distance = round(gaussian_with_variance(50, 20, athlete_id, f"dist{date_str}"), 2)
        distance = max(20, min(140, distance))
        duration = int(distance * random.gauss(8, 2))
        elevation = int(distance * random.gauss(15, 10))
        
    else:  # Other sports
        distance = round(gaussian_with_variance(8, 4, athlete_id, f"dist{date_str}"), 2)
        distance = max(2, min(25, distance))
        duration = int(random.gauss(60, 20))
        duration = max(15, min(180, duration))
        elevation = 0
    
    return distance, max(5, duration), max(0, elevation)

def generate_injury_history(athletes):
    """Generate injury_medical_history.csv"""
    injuries = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        years_training = athlete["years_training"]
        age = athlete["age"]
        comp_level = athlete["competition_level"]
        
        # Injury risk increases with years, age, and level
        base_injury_prob = 0.02 + years_training * 0.003 + age * 0.0005 + comp_level_to_num(comp_level) * 0.005
        
        # 20-30% of athletes have injuries
        if random.random() > 0.75:
            continue
        
        # Number of injuries
        num_injuries = sum(1 for _ in range(int(years_training * 12)) if random.random() < base_injury_prob)
        num_injuries = max(1, min(15, num_injuries))
        
        for _ in range(num_injuries):
            # Injury date (can be historical)
            days_ago = random.randint(0, years_training * 365)
            injury_date = END_DATE - timedelta(days=days_ago)
            injury_date_str = injury_date.strftime("%Y-%m-%d")
            
            injury_type = random.choice(INJURY_TYPES)
            
            # Severity probabilities
            severity = random.choices(
                ["Minor", "Moderate", "Severe", "Chronic"],
                weights=[0.50, 0.30, 0.15, 0.05]
            )[0]
            
            # Recovery time based on severity
            if severity == "Minor":
                recovery_days = random.randint(3, 21)
            elif severity == "Moderate":
                recovery_days = random.randint(21, 60)
            elif severity == "Severe":
                recovery_days = random.randint(60, 180)
            else:  # Chronic
                recovery_days = random.randint(90, 365)
            
            treatment = random.choice([
                "Rest", "Physical therapy", "Medical treatment", "Surgery", 
                "Medication", "Ice and elevation", "Rehabilitation program",
                "Chiropractic care", "Massage therapy", "Active recovery"
            ])
            
            recurring = "Yes" if random.random() < 0.15 else "No"
            
            # Cost estimate
            if severity == "Minor":
                cost = random.randint(0, 500)
            elif severity == "Moderate":
                cost = random.randint(300, 2000)
            elif severity == "Severe":
                cost = random.randint(1500, 10000)
            else:
                cost = random.randint(5000, 25000)
            
            injuries.append({
                "athlete_id": athlete_id,
                "injury_date": injury_date_str,
                "injury_type": injury_type,
                "severity": severity,
                "recovery_days": recovery_days,
                "treatment": treatment,
                "recurring": recurring,
                "treatment_cost_usd": cost,
                "affected_activities": 1 if severity in ["Minor", "Moderate"] else 1 if random.random() < 0.5 else 0
            })
    
    return injuries

def generate_mood_tracking(athletes):
    """Generate mood_tracking.csv"""
    mood_data = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        personality = athlete["personality_score"]
        
        # Number of entries (30-150 per athlete)
        num_entries = random.randint(30, 150)
        
        dates = sorted([START_DATE + timedelta(days=random.randint(0, DAYS_IN_PERIOD)) 
                       for _ in range(num_entries)])
        
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            
            # Base mood affected by personality
            base_mood = 5 + personality * 0.3
            
            # Daily metrics
            energy = int(gaussian_with_variance(base_mood, 1.8, athlete_id, f"energy{date_str}"))
            energy = max(1, min(10, energy))
            
            motivation = int(gaussian_with_variance(base_mood + 0.5, 1.7, athlete_id, f"motiv{date_str}"))
            motivation = max(1, min(10, motivation))
            
            stress = random.randint(1, 10)
            
            fatigue = 10 - energy + random.randint(-2, 2)
            fatigue = max(1, min(10, fatigue))
            
            soreness = random.randint(1, 10)
            
            # Overall mood correlates with other factors
            overall_mood = int((energy + motivation + (10 - stress) + (10 - fatigue)) / 4)
            overall_mood = max(1, min(10, overall_mood))
            
            # Optional notes (20% of entries)
            notes_options = [
                "", "", "", "",  # 80% empty
                "Great training session", "Feeling tired", "Good recovery day",
                "Struggled today", "Excellent workout", "Need more rest",
                "Feeling strong", "Low motivation", "Peak performance",
                "Overtrained feeling", "Refreshed and ready", "Minor discomfort"
            ]
            notes = random.choice(notes_options)
            
            mood_data.append({
                "athlete_id": athlete_id,
                "date": date_str,
                "energy_level": energy,
                "motivation_level": motivation,
                "stress_level": stress,
                "fatigue_level": fatigue,
                "soreness_level": soreness,
                "overall_mood": overall_mood,
                "mental_clarity": random.randint(1, 10),
                "notes": notes
            })
    
    return mood_data

def generate_life_timeline(athletes):
    """Generate life_timeline_career.csv"""
    timeline = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        age = athlete["age"]
        years_training = athlete["years_training"]
        comp_level = athlete["competition_level"]
        
        # 1-3 events per year
        years_active = min(years_training, age - 18)
        num_events = random.randint(years_active, years_active * 3)
        
        for _ in range(num_events):
            # Event date (can be historical)
            days_ago = random.randint(0, years_active * 365)
            event_date = END_DATE - timedelta(days=days_ago)
            event_date_str = event_date.strftime("%Y-%m-%d")
            
            event_type = random.choice(EVENT_TYPES)
            
            # Description based on type
            if event_type == "Competition":
                comp_types = ["marathon", "triathlon", "championship", "local race", "national event", 
                             "regional qualifier", "international competition"]
                description = f"Participated in {random.choice(comp_types)}"
            elif event_type == "Training milestone":
                milestones = ["100km month", "500km total", "1000km milestone", "personal best", 
                             "consistency streak", "volume record"]
                description = f"Reached {random.choice(milestones)}"
            elif event_type == "Location change":
                description = f"Relocated to new city"
            elif event_type == "Career change":
                description = f"Started new position"
            elif event_type == "Race victory":
                description = f"Won {random.choice(['local race', 'age group', 'division title'])}"
            else:
                description = f"{event_type} event"
            
            # Impact rating
            impact = int(gaussian_with_variance(6, 2, athlete_id, f"impact{event_date_str}"))
            impact = max(1, min(10, impact))
            
            timeline.append({
                "athlete_id": athlete_id,
                "event_date": event_date_str,
                "event_type": event_type,
                "description": description,
                "impact_rating": impact,
                "career_phase": comp_level
            })
    
    return timeline

def generate_social_communication(athletes):
    """Generate social_communication.csv"""
    social_data = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        network_size = athlete["social_network_size"]
        personality = athlete["personality_score"]
        
        # Communication frequency based on personality and network
        base_interactions = network_size * (0.5 + personality * 0.15)
        num_events = int(gaussian_with_variance(base_interactions, base_interactions * 0.3, athlete_id, "social"))
        num_events = max(20, min(500, num_events))
        
        dates = sorted([START_DATE + timedelta(days=random.randint(0, DAYS_IN_PERIOD)) 
                       for _ in range(num_events)])
        
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            
            # Communication type based on personality
            if personality > 6:  # Extroverted
                comm_type = random.choices(
                    ["Call", "Group chat", "Social media", "Message", "Video call"],
                    weights=[0.25, 0.25, 0.20, 0.20, 0.10]
                )[0]
            else:  # Introverted
                comm_type = random.choices(
                    ["Message", "Email", "Social media", "Call", "Group chat"],
                    weights=[0.35, 0.25, 0.20, 0.15, 0.05]
                )[0]
            
            platform = random.choice([
                "WhatsApp", "Strava", "Instagram", "Facebook", "Email", 
                "Phone", "Discord", "Telegram", "Messenger", "Slack"
            ])
            
            # Duration
            if comm_type in ["Call", "Video call"]:
                duration_min = int(gaussian_with_variance(20, 15, athlete_id, f"dur{date_str}"))
                duration_min = max(2, min(120, duration_min))
            elif comm_type == "Group chat":
                duration_min = random.randint(5, 45)
            else:
                duration_min = 0
            
            # Contact
            contact_id = f"CONTACT_{random.randint(1, network_size):04d}"
            
            # Sentiment
            sentiment = random.choices(
                ["Positive", "Neutral", "Negative", "Supportive", "Motivational"],
                weights=[0.45, 0.30, 0.10, 0.10, 0.05]
            )[0]
            
            # Topic
            topics = ["Training", "Competition", "Social", "General", "Support", 
                     "Planning", "Motivation", "Recovery", "Strategy"]
            topic = random.choice(topics)
            
            social_data.append({
                "athlete_id": athlete_id,
                "date": date_str,
                "communication_type": comm_type,
                "platform": platform,
                "duration_minutes": duration_min,
                "contact_id": contact_id,
                "sentiment": sentiment,
                "topic": topic
            })
    
    return social_data

# Generate all datasets
print("\n🔄 Generating master profiles...")
athletes = generate_master_profiles(NUM_ATHLETES)
print(f"✓ Generated {len(athletes)} athlete profiles")

# Write master profiles
with open('c:/Users/rasto/Desktop/peer/data/raw/athlete_master_profiles.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=athletes[0].keys())
    writer.writeheader()
    writer.writerows(athletes)

print("\n🔄 Generating daily health metrics...")
health_metrics = generate_daily_health_metrics(athletes)
print(f"✓ Generated {len(health_metrics)} daily health metrics")
with open('c:/Users/rasto/Desktop/peer/data/raw/daily_health_metrics.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=health_metrics[0].keys())
    writer.writeheader()
    writer.writerows(health_metrics)

print("\n🔄 Generating Strava activities...")
activities = generate_strava_activities(athletes)
print(f"✓ Generated {len(activities)} Strava activities")
with open('c:/Users/rasto/Desktop/peer/data/raw/strava_activities.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=activities[0].keys())
    writer.writeheader()
    writer.writerows(activities)

print("\n🔄 Generating injury history...")
injuries = generate_injury_history(athletes)
print(f"✓ Generated {len(injuries)} injury records")
with open('c:/Users/rasto/Desktop/peer/data/raw/injury_medical_history.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=injuries[0].keys())
    writer.writeheader()
    writer.writerows(injuries)

print("\n🔄 Generating mood tracking...")
mood_data = generate_mood_tracking(athletes)
print(f"✓ Generated {len(mood_data)} mood tracking entries")
with open('c:/Users/rasto/Desktop/peer/data/raw/mood_tracking.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=mood_data[0].keys())
    writer.writeheader()
    writer.writerows(mood_data)

print("\n🔄 Generating life timeline...")
timeline = generate_life_timeline(athletes)
print(f"✓ Generated {len(timeline)} life timeline events")
with open('c:/Users/rasto/Desktop/peer/data/raw/life_timeline_career.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=timeline[0].keys())
    writer.writeheader()
    writer.writerows(timeline)

print("\n🔄 Generating social communication...")
social_data = generate_social_communication(athletes)
print(f"✓ Generated {len(social_data)} social communication records")
with open('c:/Users/rasto/Desktop/peer/data/raw/social_communication.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=social_data[0].keys())
    writer.writeheader()
    writer.writerows(social_data)

# Summary
total_records = (len(athletes) + len(health_metrics) + len(activities) + 
                len(injuries) + len(mood_data) + len(timeline) + len(social_data))

print("\n" + "=" * 80)
print("✅ DATASET GENERATION COMPLETE")
print("=" * 80)
print(f"\n📊 Dataset Summary:")
print(f"   • Athletes: {len(athletes):,}")
print(f"   • Health metrics: {len(health_metrics):,}")
print(f"   • Activities: {len(activities):,}")
print(f"   • Injuries: {len(injuries):,}")
print(f"   • Mood entries: {len(mood_data):,}")
print(f"   • Life events: {len(timeline):,}")
print(f"   • Social interactions: {len(social_data):,}")
print(f"\n   📈 TOTAL RECORDS: {total_records:,}")
print(f"\n✅ All files saved to data/raw/")
print("=" * 80)
