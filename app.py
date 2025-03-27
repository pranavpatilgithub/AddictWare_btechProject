from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector  # Use mysql-connector-python
from flask import Flask, render_template, request, redirect, url_for, flash,session

from datetime import datetime
import pickle
import numpy as np
import mysql.connector


app = Flask(__name__)
app.secret_key = '1234'  # Change this to a secure secret key

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='myDBp',
        database='mydatabase'
    )

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/extension', methods=['GET', 'POST'])
def extension():
    return render_template('extension.html')

@app.route('/ml', methods=['GET', 'POST'])
def ml():
    return render_template('ml.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionaire():
    return render_template('questionnaire.html')

@app.route('/precautions', methods=['GET', 'POST'])
def precautions():
    return render_template('precautions.html')

@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove user from session
    session.pop('submit_count', None)
    session.pop('form_data', None)
    return redirect(url_for('home'))  # Redirect to homepage after logout


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        rpassword = request.form['rpassword']
        pet = request.form['pet']

        if password != rpassword:
            error = 'Passwords do not match!'
            return render_template('register.html', error=error)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if the user already exists
            cursor.execute("SELECT Name FROM Users WHERE Email = %s", (email,))
            if cursor.fetchone():
                error = "User already registered!"
                return render_template('register.html', error=error)

            # Hash the password before storing it
            hashed_password = generate_password_hash(password)

            # Store the current timestamp
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Insert new user into the database
            cursor.execute("INSERT INTO Users (Date, Name, Email, Password, Pet) VALUES (%s, %s, %s, %s, %s)",
                           (now, name, email, hashed_password, pet))
            conn.commit()

            # Store user session after registration
            session['user'] = name
            session['email'] = email

            return redirect(url_for('home'))

        except Exception as e:
            error = str(e)
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Fetch username and hashed password
            cursor.execute("SELECT Name, Email, Password FROM Users WHERE Email = %s", (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user[2], password):  # Verify password
                session['user'] = user[0]  # Store username in session
                session['email'] = user[1]  # Store email in session
                return redirect(url_for('home'))
            else:
                error = "Invalid Credentials! Please try again."

        except Exception as e:
            error = str(e)
        finally:
            cursor.close()
            conn.close()

    return render_template('login.html', error=error)


@app.route('/forgot', methods=['GET', 'POST'])
def forgot_password():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        pet = request.form['pet']

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Retrieve the password if email and pet name match
            cursor.execute("SELECT password FROM Users WHERE Email = %s AND pet = %s", (email, pet))
            user = cursor.fetchone()
            
            if user:
                return render_template('forgot-password.html', email=email, pet=pet, error=f"Your password is: {user[0]}")
            else:
                error = "Incorrect details! Please try again."
        
        except Exception as e:
            error = str(e)
        
        finally:
            cursor.close()
            conn.close()

    return render_template('forgot-password.html', error=error)

def generate_advanced_precautions(total_score, scores):
    """
    Generate advanced, contextualized precautions based on detailed scoring
    
    Args:
    total_score (int): Total score from the questionnaire
    scores (list): Individual question scores
    
    Returns:
    dict: Comprehensive precaution recommendations
    """
    # Analyze specific problem areas
    problem_areas = {
        'time_management': [1, 16, 17, 18],  # Questions about losing track of time
        'social_impact': [2, 3, 4, 5, 19],   # Questions about social relationships
        'productivity': [6, 8],               # Questions about work/study performance
        'emotional_dependency': [10, 12, 15, 20]  # Questions about emotional reliance on internet
    }
    
    # Detailed problem analysis
    area_scores = {
        area: sum(scores[q-1] for q in questions) 
        for area, questions in problem_areas.items()
    }
    
    # Personalized recommendations
    recommendations = {
        'Severe Addiction': {
            'summary': 'Critical Internet Dependency Detected',
            'core_recommendations': [
                'Immediate professional counseling',
                'Comprehensive digital detox program',
                'Structured daily routine with limited internet access'
            ],
            'detailed_guidance': []
        },
        'Moderate Addiction': {
            'summary': 'Significant Internet Usage Concerns',
            'core_recommendations': [
                'Structured screen time management',
                'Cognitive behavioral therapy consultation',
                'Gradual digital habit restructuring'
            ],
            'detailed_guidance': []
        },
        'Mild Addiction': {
            'summary': 'Early Stage Internet Overuse',
            'core_recommendations': [
                'Preventive self-management strategies',
                'Mindfulness and time tracking',
                'Balanced lifestyle development'
            ],
            'detailed_guidance': []
        },
        'No Addiction': {
            'summary': 'Healthy Digital Habits',
            'core_recommendations': [
                'Maintain current balanced approach',
                'Continue mindful internet usage',
                'Regular self-assessment'
            ],
            'detailed_guidance': []
        }
    }
    
    # Dynamic precaution generation
    if total_score > 79:
        current_level = 'Severe Addiction'
        recommendations[current_level]['detailed_guidance'] = [
            f"Critical Time Management Alert: Your responses suggest extreme difficulty managing online time (avg score: {area_scores['time_management']/len(problem_areas['time_management']):.2f}/5)",
            f"Social Impact Warning: Significant interference with personal relationships (avg score: {area_scores['social_impact']/len(problem_areas['social_impact']):.2f}/5)",
            "Recommended Actions:",
            "1. Seek professional digital addiction counseling",
            "2. Implement strict daily internet usage limits (max 2 hours)",
            "3. Replace online time with structured offline activities",
            "4. Consider support groups for digital addiction"
        ]
    
    elif total_score > 49:
        current_level = 'Moderate Addiction'
        recommendations[current_level]['detailed_guidance'] = [
            f"Productivity Concerns: Noticeable impact on work/study performance (avg score: {area_scores['productivity']/len(problem_areas['productivity']):.2f}/5)",
            f"Emotional Dependency Indicators: Emerging reliance on internet for emotional regulation (avg score: {area_scores['emotional_dependency']/len(problem_areas['emotional_dependency']):.2f}/5)",
            "Recommended Actions:",
            "1. Use app-based screen time tracking",
            "2. Develop alternative stress management techniques",
            "3. Create a structured daily schedule with designated offline periods",
            "4. Engage in face-to-face social interactions"
        ]
    
    elif total_score > 30:
        current_level = 'Mild Addiction'
        recommendations[current_level]['detailed_guidance'] = [
            f"Early Warning: Potential developing internet dependency (time management avg: {area_scores['time_management']/len(problem_areas['time_management']):.2f}/5)",
            "Recommended Preventive Strategies:",
            "1. Implement daily screen time limits",
            "2. Practice mindful internet usage",
            "3. Develop diverse offline interests",
            "4. Regular digital habit self-assessment"
        ]
    
    else:
        current_level = 'No Addiction'
        recommendations[current_level]['detailed_guidance'] = [
            "Maintain your current balanced approach to internet usage",
            "Continue practicing healthy digital habits",
            "Stay aware of potential future dependency risks"
        ]
    
    return {
        'level': current_level,
        'total_score': total_score,
        'area_scores': area_scores,
        'recommendations': recommendations[current_level]
    }

name = ""
result = ""
qresult = ""
total_score = 0

@app.route('/answers', methods=['GET', 'POST'])
def answers():
    # Ensure the user is logged in
    if 'user' not in session:
        flash("You must be logged in to access this page.", "danger")
        return redirect(url_for('login'))

    email = session.get('email')  # Fetch currently logged-in user's email
    
    
    
    # Initialize submit_count if it doesn't exist
    if 'submit_count' not in session:
        session['submit_count'] = 0
    
    disable_submit = False

    if request.method == 'POST':
        # Get the form scores
        scores = [int(request.form.get(f'q{i}', 0)) for i in range(1, 21)]
        total_score = sum(scores)
        average_score = total_score / len(scores) if scores else 0

        # Generate advanced precautions (Ensure this function is defined in your app)
        advanced_precautions = generate_advanced_precautions(total_score, scores)

        # Determine addiction level
        if total_score > 79:
            qresult = 'Extreme Addiction'
        elif total_score > 49:
            qresult = 'Severe Addiction'
        elif total_score > 30:
            qresult = 'Moderate Addiction'
        else:
            qresult = 'No Addiction'

        # Combine recommendations into a single precaution string
        precautions = " ".join(advanced_precautions.get('recommendations', {}).get('core_recommendations', []))

        # Store the form data in the session after the first or second submission
        if session['submit_count'] < 2:
            # Store form data in the session
            session['form_data'] = {
                'scores': scores,
                'total_score': total_score,
                'average_score': average_score,
                'result': qresult,
                'precaution': precautions,
                'advanced_recommendations': advanced_precautions
            }
            # Increment the submission count
            session['submit_count'] += 1
            
        print(session['form_data'])
        print(session['submit_count'])

        if session['submit_count'] >= 2:
            disable_submit = True


        # After the second submission, you can store the data in a permanent storage (like DB)
        # if session['submit_count'] == 2:
        #     # Save data in the database or elsewhere as needed
        #     # Uncomment and implement the database connection if needed
        #     # try:
        #     #     conn = get_db_connection()
        #     #     cursor = conn.cursor()
        #     #     cursor.execute("INSERT INTO records (email, questionnaire_result) VALUES (%s, %s)", (email, qresult))
        #     #     conn.commit()
        #     # except Exception as e:
        #     #     conn.rollback()
        #     #     flash("Error saving data!", "danger")
        #     #     print("Database error:", e)
        #     # finally:
        #     #     cursor.close()
        #     #     conn.close()

        #     return render_template('thank_you.html') 

        # If it's the first or second submission, render the result on the questionnaire page
        return render_template('questionnaire.html',
                               name=session.get('user'),  # Use 'user' as per session setup
                               total_score=total_score,
                               average_score=average_score,
                               result=qresult,
                               precaution=precautions,
                               advanced_recommendations=advanced_precautions)

    return render_template('questionnaire.html', name=session.get('user'), disable_submit=disable_submit)




# Load trained models
dt_model = pickle.load(open("decision_tree.pkl", "rb"))
rf_model = pickle.load(open("random_forest.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure the user is logged in
        if 'user' not in session:
            flash("You must be logged in to access this page.", "danger")
            return redirect(url_for('login'))

        email = session.get('email')  # Fetch currently logged-in user's email
    
        # Ensure submit_count exists in session
        if 'submit_count_t' not in session:
            session['submit_count_t'] = 0

        # Initialize disable_submit flag
        disable_submitt = False

        # Extract 10 input values from form
        q = [int(request.form[f'q{i}']) for i in range(1, 11)]

        # Generate features
        FBhours = q[0] + q[6]
        FBA = sum(q) / 10
        FPS = q[4]
        FRS = q[3]
        FUS = q[0] + q[8]
        FSS = q[8]
        FSTS = q[9]
        INAD = int(sum(q) > 20)  # Binary addiction classification
        FAD = q[0] + q[1] + q[5]

        # Create feature array
        features_array = np.array([FBhours, FBA, FPS, FRS, FUS, FSS, FSTS, INAD, FAD]).reshape(1, -1)

        # Predict using both models
        dt_prediction = dt_model.predict(features_array)[0]
        rf_prediction = rf_model.predict(features_array)[0]

        # Hybrid Prediction Logic
        if dt_prediction == 'none':
            hybrid_prediction = dt_prediction  # If DT predicts None, hybrid should also be None
        elif dt_prediction == rf_prediction:
            hybrid_prediction = dt_prediction  # If both agree, use that prediction
        else:
            hybrid_prediction = rf_prediction  # Prefer RF in case of conflict

        # Store the form data in session after the first or second submission
        if session['submit_count_t'] < 2:
            session['form_data_2'] = {
                'dt_prediction': dt_prediction,
                'rf_prediction': rf_prediction,
                'hybrid_prediction': hybrid_prediction
            }
            # Increment submit count
            session['submit_count_t'] += 1
        
        
        print(session['form_data_2'])
        # print(session['submit_count'])

        # Disable submit after 2 submissions
        if session['submit_count_t'] >= 2:
            disable_submitt = True

        # Return the result and store it in session for future use
        return render_template('ml.html', 
                               n=dt_prediction, 
                               m=rf_prediction, 
                               p=hybrid_prediction,
                               disable_submit=disable_submitt)

    except Exception as e:
        return str(e)  # Returns the error message for debugging

if __name__ == '__main__':
    app.run(debug=True)

