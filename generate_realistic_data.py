"""
Generate realistic, internally consistent CSV datasets for athlete tracking system.
Creates 150 athletes with believable noise, natural variance, and cross-file consistency.
"""

import random
import csv
from datetime import datetime, timedelta
import math

# Set seed for reproducibility but with enough variance
random.seed(42)

# Configuration
NUM_ATHLETES = 150
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Realistic data ranges
FIRST_NAMES_MALE = ["Alex", "James", "Michael", "David", "Chris", "Ryan", "Daniel", "Matthew", "Kevin", "Thomas",
                    "Carlos", "Marco", "Lucas", "Diego", "Chen", "Raj", "Omar", "Hans", "Erik", "Luca",
                    "Antonio", "Rafael", "Hassan", "Mohammed", "Yuki", "Kai", "Felix", "Sebastian", "Viktor"]
FIRST_NAMES_FEMALE = ["Sarah", "Emma", "Maria", "Anna", "Sophie", "Olivia", "Emily", "Charlotte", "Grace", "Lily",
                      "Sofia", "Isabella", "Elena", "Nina", "Maya", "Zara", "Clara", "Rosa", "Eva", "Luna",
                      "Aria", "Stella", "Mia", "Chloe", "Nora", "Amelia", "Hannah", "Freya", "Priya", "Sakura"]
LAST_NAMES = ["Smith", "Johnson", "Garcia", "Martinez", "Rodriguez", "Wilson", "Anderson", "Taylor", "Moore", "Brown",
              "Chen", "Patel", "Kumar", "Kim", "Lee", "Wang", "Schmidt", "Müller", "Silva", "Santos",
              "Dubois", "Martin", "Bernard", "Larsson", "Andersen", "Kowalski", "Novak", "Petrov", "Ivanov", "Costa"]

CITIES_COUNTRIES = [
    ("New York, USA", "USA", -4, 40.7), ("Los Angeles, USA", "USA", -8, 34.0), ("Chicago, USA", "USA", -6, 41.8),
    ("London, UK", "UK", 0, 51.5), ("Manchester, UK", "UK", 0, 53.4), ("Berlin, Germany", "Germany", 1, 52.5),
    ("Munich, Germany", "Germany", 1, 48.1), ("Paris, France", "France", 1, 48.8), ("Lyon, France", "France", 1, 45.7),
    ("Madrid, Spain", "Spain", 1, 40.4), ("Barcelona, Spain", "Spain", 1, 41.3), ("Rome, Italy", "Italy", 1, 41.9),
    ("Milan, Italy", "Italy", 1, 45.4), ("Amsterdam, Netherlands", "Netherlands", 1, 52.3),
    ("Stockholm, Sweden", "Sweden", 1, 59.3), ("Copenhagen, Denmark", "Denmark", 1, 55.6),
    ("Oslo, Norway", "Norway", 1, 59.9), ("Helsinki, Finland", "Finland", 2, 60.1),
    ("Tokyo, Japan", "Japan", 9, 35.6), ("Seoul, South Korea", "South Korea", 9, 37.5),
    ("Beijing, China", "China", 8, 39.9), ("Shanghai, China", "China", 8, 31.2),
    ("Mumbai, India", "India", 5.5, 19.0), ("Delhi, India", "India", 5.5, 28.6),
    ("Sydney, Australia", "Australia", 10, -33.8), ("Melbourne, Australia", "Australia", 10, -37.8),
    ("Toronto, Canada", "Canada", -5, 43.6), ("Vancouver, Canada", "Canada", -8, 49.2),
    ("São Paulo, Brazil", "Brazil", -3, -23.5), ("Rio de Janeiro, Brazil", "Brazil", -3, -22.9),
    ("Mexico City, Mexico", "Mexico", -6, 19.4), ("Buenos Aires, Argentina", "Argentina", -3, -34.6),
    ("Moscow, Russia", "Russia", 3, 55.7), ("Dubai, UAE", "UAE", 4, 25.2),
    ("Singapore, Singapore", "Singapore", 8, 1.3), ("Bangkok, Thailand", "Thailand", 7, 13.7)
]

SPORTS = ["Running", "Cycling", "Swimming", "Triathlon", "Football", "Basketball", "Tennis", "Rowing", "CrossFit"]
COMPETITION_LEVELS = ["Beginner", "Intermediate", "Advanced", "Elite Amateur"]
COMM_PREFS = ["Text-focused", "Voice-focused", "Mixed", "App-based"]

def generate_master_profiles(num_athletes):
    """Generate detailed_athlete_master_profiles.csv"""
    athletes = []
    
    for i in range(1, num_athletes + 1):
        athlete_id = f"ATH_{i:03d}"
        gender = random.choice(["Male", "Female"])
        age = int(random.gauss(30, 8))  # Normal distribution around 30
        age = max(18, min(60, age))  # Clamp to 18-60
        
        # Name based on gender
        first_name = random.choice(FIRST_NAMES_MALE if gender == "Male" else FIRST_NAMES_FEMALE)
        last_name = random.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"
        
        # Location
        location, nationality, tz_offset, lat = random.choice(CITIES_COUNTRIES)
        
        # Physical attributes with realistic variance
        if gender == "Male":
            height = int(random.gauss(178, 7))  # cm
            weight = int(random.gauss(75, 10))  # kg
        else:
            height = int(random.gauss(165, 6))  # cm
            weight = int(random.gauss(62, 8))  # kg
        height = max(150, min(200, height))
        weight = max(45, min(120, weight))
        
        # Sport and experience
        sport = random.choice(SPORTS)
        comp_level = random.choice(COMPETITION_LEVELS)
        
        # Years training correlates with age and level
        level_years = {"Beginner": (1, 3), "Intermediate": (3, 8), "Advanced": (6, 15), "Elite Amateur": (10, 20)}
        min_years, max_years = level_years[comp_level]
        max_possible = max(min_years, min(max_years, age - 18))
        years_training = random.randint(min_years, max_possible)
        
        # Social attributes
        social_network = int(random.gauss(50, 25))
        social_network = max(5, min(150, social_network))
        
        coaching = 1 if random.random() < 0.6 else 0  # 60% have coaching
        partners = random.randint(0, 7)
        
        comm_pref = random.choice(COMM_PREFS)
        career_status = "Active" if random.random() < 0.95 else "Retired"
        
        athletes.append({
            "athlete_id": athlete_id,
            "name": name,
            "age": age,
            "gender": gender,
            "location": location,
            "nationality": nationality,
            "primary_sport": sport,
            "competition_level": comp_level,
            "years_training": years_training,
            "height_cm": height,
            "weight_kg": weight,
            "social_network_size": social_network,
            "coaching_support": coaching,
            "training_partners": partners,
            "career_status": career_status,
            "communication_preference": comm_pref
        })
    
    return athletes

def generate_daily_health_metrics(athletes):
    """Generate detailed_daily_health_metrics.csv"""
    metrics = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        age = athlete["age"]
        gender = athlete["gender"]
        comp_level = athlete["competition_level"]
        
        # Base metrics influenced by age and level
        base_resting_hr = 60 if comp_level in ["Advanced", "Elite Amateur"] else 70
        base_resting_hr += (age - 30) * 0.3  # Slightly increases with age
        
        # Generate 50-150 days of data per athlete
        num_days = random.randint(50, 150)
        
        for _ in range(num_days):
            date = START_DATE + timedelta(days=random.randint(0, (END_DATE - START_DATE).days))
            date_str = date.strftime("%Y-%m-%d")
            
            # Daily metrics with realistic variance
            resting_hr = max(45, int(random.gauss(base_resting_hr, 5)))
            recovery_score = max(30, min(100, int(random.gauss(75, 12))))
            sleep_hours = max(4.0, min(10.0, round(random.gauss(7.2, 1.0), 1)))
            steps = max(1000, int(random.gauss(9000, 3000)))
            hrv = max(20, int(random.gauss(60, 15)))  # Heart rate variability
            
            # Stress and mood correlate slightly
            stress_level = random.randint(1, 10)
            mood = max(1, min(10, int(random.gauss(7, 2))))
            
            # Weight varies slightly day to day
            weight = athlete["weight_kg"] + round(random.gauss(0, 0.5), 1)
            
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
                "weight_kg": weight
            })
    
    return metrics

def generate_strava_activities(athletes):
    """Generate detailed_strava_activities.csv"""
    activities = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        sport = athlete["primary_sport"]
        comp_level = athlete["competition_level"]
        age = athlete["age"]
        
        # Activity frequency based on level
        level_activities = {"Beginner": (30, 80), "Intermediate": (80, 180), 
                          "Advanced": (150, 300), "Elite Amateur": (250, 400)}
        min_act, max_act = level_activities[comp_level]
        num_activities = random.randint(min_act, max_act)
        
        for _ in range(num_activities):
            date = START_DATE + timedelta(days=random.randint(0, (END_DATE - START_DATE).days))
            date_str = date.strftime("%Y-%m-%d")
            
            # Activity type aligns with primary sport but with variety
            if random.random() < 0.75:
                activity_type = sport
            else:
                activity_type = random.choice([s for s in SPORTS if s != sport])
            
            # Distance based on sport and level
            if activity_type in ["Running", "Cycling"]:
                if activity_type == "Running":
                    base_dist = 8 if comp_level in ["Beginner"] else 12 if comp_level == "Intermediate" else 15
                    distance = max(3, round(random.gauss(base_dist, 4), 2))
                else:  # Cycling
                    base_dist = 30 if comp_level == "Beginner" else 50 if comp_level == "Intermediate" else 70
                    distance = max(10, round(random.gauss(base_dist, 15), 2))
            elif activity_type == "Swimming":
                distance = max(0.5, round(random.gauss(2.5, 1.0), 2))
            else:
                distance = max(5, round(random.gauss(10, 5), 2))
            
            # Duration and pace
            if activity_type == "Running":
                pace = 5.5 + (40 - age) * 0.02  # min/km, faster when younger
                duration = distance * pace
            elif activity_type == "Cycling":
                speed = 25 + (40 - age) * 0.1  # km/h
                duration = (distance / speed) * 60
            else:
                duration = distance * random.gauss(8, 2)
            
            duration = max(10, int(duration))  # minutes
            
            # Elevation for outdoor activities
            if activity_type in ["Running", "Cycling"]:
                elevation = max(0, int(distance * random.gauss(15, 10)))
            else:
                elevation = 0
            
            # Heart rate
            max_hr = 220 - age
            avg_hr = int(max_hr * random.gauss(0.70, 0.08))  # 60-80% of max
            avg_hr = max(100, min(int(max_hr * 0.9), avg_hr))
            max_hr_activity = min(int(max_hr * 0.95), avg_hr + random.randint(10, 25))
            
            # Calories
            calories = int(duration * random.gauss(8, 2))
            
            activities.append({
                "athlete_id": athlete_id,
                "date": date_str,
                "activity_type": activity_type,
                "distance_km": distance,
                "duration_minutes": duration,
                "elevation_gain_m": elevation,
                "avg_heart_rate": avg_hr,
                "max_heart_rate": max_hr_activity,
                "calories_burned": calories
            })
    
    return activities

def generate_injury_history(athletes):
    """Generate detailed_injury_medical_history.csv"""
    injuries = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        years_training = athlete["years_training"]
        age = athlete["age"]
        
        # Injury probability increases with years and age
        injury_rate = min(0.3, 0.03 * years_training + 0.002 * age)
        num_injuries = sum(1 for _ in range(int(years_training * 12)) if random.random() < injury_rate)
        
        if num_injuries == 0 and random.random() < 0.3:  # Some athletes have zero injuries
            continue
        
        injury_types = ["Muscle strain", "Knee pain", "Ankle sprain", "Lower back pain", 
                       "Shin splints", "Tendonitis", "Stress fracture", "IT band syndrome",
                       "Plantar fasciitis", "Shoulder pain"]
        
        for _ in range(num_injuries):
            date = START_DATE + timedelta(days=random.randint(-365*years_training, 0))
            date_str = date.strftime("%Y-%m-%d")
            
            injury_type = random.choice(injury_types)
            severity = random.choice(["Minor", "Minor", "Minor", "Moderate", "Moderate", "Severe"])
            
            # Recovery time based on severity
            if severity == "Minor":
                recovery_days = random.randint(5, 21)
            elif severity == "Moderate":
                recovery_days = random.randint(21, 60)
            else:
                recovery_days = random.randint(60, 180)
            
            treatment = random.choice(["Rest", "Physical therapy", "Medical treatment", 
                                      "Surgery", "Medication", "Ice and elevation"])
            
            injuries.append({
                "athlete_id": athlete_id,
                "injury_date": date_str,
                "injury_type": injury_type,
                "severity": severity,
                "recovery_days": recovery_days,
                "treatment": treatment,
                "recurring": "Yes" if random.random() < 0.2 else "No"
            })
    
    return injuries

def generate_mood_tracking(athletes):
    """Generate detailed_mood_tracking.csv"""
    mood_data = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        
        # 30-120 mood entries per athlete
        num_entries = random.randint(30, 120)
        
        for _ in range(num_entries):
            date = START_DATE + timedelta(days=random.randint(0, (END_DATE - START_DATE).days))
            date_str = date.strftime("%Y-%m-%d")
            
            # Mood metrics
            energy = random.randint(1, 10)
            motivation = random.randint(1, 10)
            stress = random.randint(1, 10)
            overall_mood = max(1, min(10, int(random.gauss(7, 2))))
            
            # Correlation: high stress -> lower mood
            if stress > 7:
                overall_mood = max(1, overall_mood - random.randint(1, 3))
            
            # Notes occasionally present
            notes = "" if random.random() < 0.7 else random.choice([
                "Felt great today", "Tired from yesterday", "Good training session",
                "Struggled with motivation", "Recovery day needed", "Feeling strong"
            ])
            
            mood_data.append({
                "athlete_id": athlete_id,
                "date": date_str,
                "energy_level": energy,
                "motivation_level": motivation,
                "stress_level": stress,
                "overall_mood": overall_mood,
                "notes": notes
            })
    
    return mood_data

def generate_life_timeline(athletes):
    """Generate detailed_life_timeline_career.csv"""
    timeline = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        age = athlete["age"]
        years_training = athlete["years_training"]
        
        # Life events per athlete
        num_events = random.randint(3, 12)
        
        event_types = ["Competition", "Training milestone", "Location change", "Career change",
                      "Injury recovery", "Personal achievement", "Education", "Family event"]
        
        for _ in range(num_events):
            # Events distributed over their lifetime
            years_ago = random.randint(0, min(age - 18, 20))
            date = datetime.now() - timedelta(days=years_ago * 365 + random.randint(0, 365))
            date_str = date.strftime("%Y-%m-%d")
            
            event_type = random.choice(event_types)
            
            if event_type == "Competition":
                description = f"Participated in {random.choice(['marathon', 'triathlon', 'championship', 'local race'])}"
            elif event_type == "Training milestone":
                description = f"Reached {random.choice(['100km', '500km', '1000km', 'personal best'])}"
            elif event_type == "Location change":
                description = f"Moved to new city"
            elif event_type == "Career change":
                description = f"Started new job"
            else:
                description = f"{event_type} event"
            
            timeline.append({
                "athlete_id": athlete_id,
                "event_date": date_str,
                "event_type": event_type,
                "description": description,
                "impact_rating": random.randint(1, 10)
            })
    
    return timeline

def generate_social_communication(athletes):
    """Generate detailed_social_communication.csv"""
    social_data = []
    
    for athlete in athletes:
        athlete_id = athlete["athlete_id"]
        network_size = athlete["social_network_size"]
        
        # Communication events based on network size
        num_events = int(network_size * random.gauss(2, 0.5))
        num_events = max(10, min(300, num_events))
        
        for _ in range(num_events):
            date = START_DATE + timedelta(days=random.randint(0, (END_DATE - START_DATE).days))
            date_str = date.strftime("%Y-%m-%d")
            
            comm_type = random.choice(["Message", "Call", "Group chat", "Social media", "Email"])
            platform = random.choice(["WhatsApp", "Strava", "Instagram", "Email", "Phone", "Discord"])
            
            duration_min = 0
            if comm_type in ["Call", "Group chat"]:
                duration_min = random.randint(5, 60)
            
            contact_id = f"CONTACT_{random.randint(1, network_size):03d}"
            
            sentiment = random.choice(["Positive", "Positive", "Positive", "Neutral", "Neutral", "Negative"])
            
            social_data.append({
                "athlete_id": athlete_id,
                "date": date_str,
                "communication_type": comm_type,
                "platform": platform,
                "duration_minutes": duration_min,
                "contact_id": contact_id,
                "sentiment": sentiment
            })
    
    return social_data

# Generate all datasets
print("Generating realistic athlete datasets...")
print("=" * 60)

athletes = generate_master_profiles(NUM_ATHLETES)
print(f"✓ Generated {len(athletes)} athlete profiles")

# Write master profiles
with open('c:/Users/rasto/Desktop/peer/data/raw/detailed_athlete_master_profiles.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=athletes[0].keys())
    writer.writeheader()
    writer.writerows(athletes)

# Generate and write health metrics
health_metrics = generate_daily_health_metrics(athletes)
print(f"✓ Generated {len(health_metrics)} daily health metrics")
with open('c:/Users/rasto/Desktop/peer/data/raw/detailed_daily_health_metrics.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=health_metrics[0].keys())
    writer.writeheader()
    writer.writerows(health_metrics)

# Generate and write activities
activities = generate_strava_activities(athletes)
print(f"✓ Generated {len(activities)} Strava activities")
with open('c:/Users/rasto/Desktop/peer/data/raw/detailed_strava_activities.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=activities[0].keys())
    writer.writeheader()
    writer.writerows(activities)

# Generate and write injuries
injuries = generate_injury_history(athletes)
print(f"✓ Generated {len(injuries)} injury records")
with open('c:/Users/rasto/Desktop/peer/data/raw/detailed_injury_medical_history.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=injuries[0].keys())
    writer.writeheader()
    writer.writerows(injuries)

# Generate and write mood tracking
mood_data = generate_mood_tracking(athletes)
print(f"✓ Generated {len(mood_data)} mood tracking entries")
with open('c:/Users/rasto/Desktop/peer/data/raw/detailed_mood_tracking.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=mood_data[0].keys())
    writer.writeheader()
    writer.writerows(mood_data)

# Generate and write life timeline
timeline = generate_life_timeline(athletes)
print(f"✓ Generated {len(timeline)} life timeline events")
with open('c:/Users/rasto/Desktop/peer/data/raw/detailed_life_timeline_career.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=timeline[0].keys())
    writer.writeheader()
    writer.writerows(timeline)

# Generate and write social communication
social_data = generate_social_communication(athletes)
print(f"✓ Generated {len(social_data)} social communication records")
with open('c:/Users/rasto/Desktop/peer/data/raw/detailed_social_communication.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=social_data[0].keys())
    writer.writeheader()
    writer.writerows(social_data)

print("=" * 60)
print("✅ All datasets generated successfully!")
print(f"\nDataset summary:")
print(f"  - Athletes: {len(athletes)}")
print(f"  - Health metrics: {len(health_metrics)}")
print(f"  - Activities: {len(activities)}")
print(f"  - Injuries: {len(injuries)}")
print(f"  - Mood entries: {len(mood_data)}")
print(f"  - Life events: {len(timeline)}")
print(f"  - Social interactions: {len(social_data)}")
print(f"\n✅ Data generation complete with realistic variance and cross-file consistency!")
