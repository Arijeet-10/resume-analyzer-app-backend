# api.py

import os
import io
import datetime
import time
import random
import traceback

# Flask imports for creating the web server and handling requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# Imports for resume analysis
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
import nltk

# Import your data lists from the Courses.py file
# Make sure Courses.py is in the same directory as api.py
try:
    from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
except ImportError:
    # Provide default empty lists if Courses.py is not found, to prevent crashes
    ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos = [], [], [], [], [], [], []
    print("WARNING: Courses.py not found. All recommendations will be empty.")


# Download necessary NLTK data package if it doesn't exist
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("Download complete.")


# --- Helper Function to read PDF text ---
def pdf_reader(file_path):
    """
    Extracts raw text from a PDF file.
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

# --- Helper Function to recommend courses ---
def course_recommender(course_list):
    """
    Recommends a shuffled list of courses.
    """
    random.shuffle(course_list)
    recommended_courses = []
    # Recommend a fixed number of courses, e.g., 5
    for c_name, c_link in course_list[:5]:
        recommended_courses.append({"name": c_name, "link": c_link})
    return recommended_courses

# --- Main Analysis Function ---
def analyze_resume(resume_path):
    """
    Processes the resume file and returns a dictionary of analysis results.
    """
    # Parse the resume using pyresparser
    try:
        resume_data = ResumeParser(resume_path).get_extracted_data()
    except Exception as e:
         return {"error": f"Error parsing resume with pyresparser: {e}"}

    if not resume_data:
        return {"error": "Could not extract any data from the resume. Is it a valid resume format?"}

    # Get the full text for keyword-based section checking
    resume_text = pdf_reader(resume_path)
    resume_text_lower = resume_text.lower()

    # 1. Determine Candidate Level based on resume length
    cand_level = ''
    page_count = resume_data.get('no_of_pages', 0)
    if page_count == 1:
        cand_level = "Fresher"
    elif page_count == 2:
        cand_level = "Intermediate"
    elif page_count >= 3:
        cand_level = "Experienced"

    # 2. Field and Skill Recommendation
    skills_lower = {skill.lower() for skill in resume_data.get('skills', [])}
    
    # Keyword lists for different fields
    ds_keyword = ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit', 'python']
    web_keyword = ['react', 'django', 'node js', 'react js', 'php', 'laravel', 'magento', 'wordpress', 'javascript', 'angular js', 'c#', 'flask']
    android_keyword = ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy', 'java']
    ios_keyword = ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode']
    uiux_keyword = ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframe', 'photoshop', 'illustrator']
    
    reco_field = 'Other'
    recommended_skills = []
    recommended_courses = []

    # Match skills to fields
    if any(i in ds_keyword for i in skills_lower):
        reco_field = 'Data Science'
        recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining', 'Clustering & Classification', 'Data Analytics', 'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras', 'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', 'Flask', 'Streamlit']
        recommended_courses = course_recommender(ds_course)
    elif any(i in web_keyword for i in skills_lower):
        reco_field = 'Web Development'
        recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento', 'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
        recommended_courses = course_recommender(web_course)
    elif any(i in android_keyword for i in skills_lower):
        reco_field = 'Android Development'
        recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java', 'Kivy', 'GIT', 'SDK', 'SQLite']
        recommended_courses = course_recommender(android_course)
    elif any(i in ios_keyword for i in skills_lower):
        reco_field = 'IOS Development'
        recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode', 'Objective-C', 'SQLite', 'Plist', 'StoreKit',"UI-Kit",'AV Foundation','Auto-Layout']
        recommended_courses = course_recommender(ios_course)
    elif any(i in uiux_keyword for i in skills_lower):
        reco_field = 'UI-UX Development'
        recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign']
        recommended_courses = course_recommender(uiux_course)

    # 3. Resume Score Calculation based on section presence
    resume_score = 0
    score_breakdown = {}
    
    sections_to_check = {
        'objective': ['objective', 'summary'],
        'declaration': ['declaration'],
        'hobbies': ['hobbies', 'interests'],
        'achievements': ['achievements', 'awards', 'certifications'],
        'projects': ['projects', 'experience', 'work experience']
    }

    for section, keywords in sections_to_check.items():
        if any(keyword in resume_text_lower for keyword in keywords):
            resume_score += 20
            score_breakdown[section] = True
        else:
            score_breakdown[section] = False

    # 4. Bonus Video Recommendations
    resume_video_rec = random.choice(resume_videos) if resume_videos else None
    interview_video_rec = random.choice(interview_videos) if interview_videos else None

    # 5. Compile all results into a final dictionary
    analysis_result = {
        "name": resume_data.get('name'),
        "email": resume_data.get('email'),
        "mobile_number": resume_data.get('mobile_number'),
        "skills": resume_data.get('skills'),
        "no_of_pages": page_count,
        "candidate_level": cand_level,
        "predicted_field": reco_field,
        "recommended_skills": recommended_skills,
        "recommended_courses": recommended_courses,
        "resume_score": resume_score,
        "score_breakdown": score_breakdown,
        "resume_video_rec": resume_video_rec,
        "interview_video_rec": interview_video_rec,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return analysis_result

# --- Flask App Setup ---
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing to allow the frontend to call this API
CORS(app) 

# Configure a folder to temporarily store uploaded resumes
UPLOAD_FOLDER = 'Uploaded_Resumes_API'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- API Endpoint Definition ---
@app.route('/analyze', methods=['POST'])
def analyze_route():
    """
    Flask route to handle the resume upload and analysis request.
    """
    # Check if a file was sent in the request
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file part in the request"}), 400
    
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    file_path = None
    if file:
        # Create a unique filename to avoid conflicts and save the file
        filename = f"{time.time()}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # This block ensures the code continues even if analysis fails,
        # and that the temporary file is always cleaned up.
        try:
            print(f"Analyzing {filename}...")
            analysis_data = analyze_resume(file_path)
            
            # Check if the analysis function itself returned an error
            if "error" in analysis_data:
                print(f"Analysis error for {filename}: {analysis_data['error']}")
                return jsonify(analysis_data), 400
            
            print(f"Analysis successful for {filename}.")
            return jsonify(analysis_data)
            
        except Exception as e:
            # If any unexpected error occurs, log it and return a generic server error
            print("--- AN UNEXPECTED ERROR OCCURRED ---")
            traceback.print_exc()
            print("------------------------------------")
            return jsonify({"error": "An unexpected server error occurred during analysis."}), 500
            
        finally:
            # This 'finally' block ensures the uploaded file is always deleted
            # after processing, whether it was successful or not.
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up {filename}.")
    
    return jsonify({"error": "Invalid file provided."}), 400


# Run the Flask app
if __name__ == '__main__':
    # Running in debug mode provides helpful error messages
    app.run(debug=True, port=5001)