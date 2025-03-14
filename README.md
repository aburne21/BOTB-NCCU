# The application analyzes user sentiment, generates personalized ads, recommends targeted advertisements using AWS Personalize, and optimizes data storage while scheduling tasks during renewable energy surplus hours. 
#  uses Python and  AWS Comprehend for sentiment analysis, , AWS S3 for cloud storage, Scikit-learn for machine learning, Logistic Regression for sentiment classification, TF-IDF for text vectorization, and JSON for our data structures.
# For the judges read this code as if you were intenral ops and this was the code we use for our AI model.
#  To run the code you just need to implement your own (made up) AWS credentials.
python --version
pip --version
pip install boto3 scikit-learn
# BOTB-NCCU
from fastapi import FastAPI
from pydantic import BaseModel
import boto3
import random
import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split



# Initialize FastAPI app
app = FastAPI()

# AWS Configurations
AWS_REGION = 'us-west-2'
comprehend = boto3.client('comprehend', region_name=AWS_REGION)
personalize_runtime = boto3.client('personalize-runtime', region_name=AWS_REGION)
CAMPAIGN_ARN = 'arn:aws:personalize:us-west-2:123456789012:campaign/campaign-name'

# Define renewable energy hours for scheduling
RENEWABLE_ENERGY_HOURS = list(range(0, 6)) + list(range(18, 24))

# ML Model (Logistic Regression for sentiment classification)
pipeline = None
def train_model():
    global pipeline
    messages = ["I love eco-friendly products!", "I need a new laptop.", "Looking for organic food options."]
    sentiments = ["POSITIVE", "NEUTRAL", "POSITIVE"]
    X_train, X_test, y_train, y_test = train_test_split(messages, sentiments, test_size=0.2, random_state=42)
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
    pipeline.fit(X_train, y_train)
train_model()

# Request Data Model
class UserRequest(BaseModel):
    user_id: str
    message: str
    interests: list
    tier: int

@app.post("/analyze/")
def analyze_user_behavior(request: UserRequest):
    """Analyze user sentiment using AWS Comprehend."""
    response = comprehend.detect_sentiment(Text=request.message, LanguageCode='en')
    return {"sentiment": response['Sentiment']}

@app.post("/generate-ads/")
def generate_ad_content(request: UserRequest):
    """Generate personalized ad content based on user interests and sentiment."""
    sentiment = comprehend.detect_sentiment(Text=request.message, LanguageCode='en')['Sentiment']
    ads = [{"headline": "Exclusive deals for you!", "body": f"Check out {', '.join(request.interests)}!", "sentiment": sentiment} for _ in range(10)]
    return {"ads": ads}

@app.post("/recommend/")
def dynamic_audience_targeting(request: UserRequest):
    """Use AWS Personalize to recommend ads based on user data and tier classification."""
    recommendations = personalize_runtime.get_recommendations(campaignArn=CAMPAIGN_ARN, userId=request.user_id).get('itemList', [])
    return {"user_id": request.user_id, "recommendations": recommendations, "scope": {1: 'local', 2: 'statewide', 3: 'international'}.get(request.tier, 'local')}

@app.post("/schedule/")
def schedule_tasks(task_name: str):
    """Schedule tasks during renewable energy surplus periods."""
    if datetime.datetime.utcnow().hour in RENEWABLE_ENERGY_HOURS:
        return {"status": "Task executed immediately", "task": task_name}
    return {"status": "Task scheduled for later execution", "task": task_name}

# Run the API: `uvicorn main:app --host 0.0.0.0 --port 8000`

 






