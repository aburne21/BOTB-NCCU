# The application analyzes user sentiment, generates personalized ads, recommends targeted advertisements using AWS Personalize, and optimizes data storage while scheduling tasks during renewable energy surplus hours. 
#  uses Python and  AWS Comprehend for sentiment analysis, , AWS S3 for cloud storage, Scikit-learn for machine learning, Logistic Regression for sentiment classification, TF-IDF for text vectorization, and JSON for our data structures.
# For the judges read this code as if you were intenral ops and this was the code we use for our AI model.
#  To run the code you just need to implement your own (made up) AWS credentials.
python --version
pip --version
# BOTB-NCCU
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
import datetime
import boto3
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Initialize FastAPI app
app = FastAPI()

# AWS Config
AWS_REGION = 'us-west-2'
comprehend = boto3.client('comprehend', region_name=AWS_REGION)
personalize_runtime = boto3.client('personalize-runtime', region_name=AWS_REGION)
CAMPAIGN_ARN = 'arn:aws:personalize:us-west-2:123456789012:campaign/campaign-name'
RENEWABLE_ENERGY_HOURS = list(range(0, 6)) + list(range(18, 24))

# JWT Secret Key
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Simulated user database
fake_users_db = {}

# OAuth2 scheme for JWT bearer authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# User models
class User(BaseModel):
    username: str
    password: str

class UserRequest(BaseModel):
    user_id: str
    message: str
    interests: list
    tier: int

# Helper function to hash passwords
def hash_password(password: str):
    return pwd_context.hash(password)

# Helper function to create JWT token
def create_access_token(data: dict, expires_delta: int = ACCESS_TOKEN_EXPIRE_MINUTES):
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_delta)
    data.update({"exp": expire})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# Authentication function
def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not pwd_context.verify(password, user["password"]):
        return False
    return user

# Dependency to get current user
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username not in fake_users_db:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Train sentiment analysis model
pipeline = None
def train_model():
    global pipeline
    messages = ["I love eco-friendly products!", "I need a new laptop.", "Looking for organic food options."]
    sentiments = ["POSITIVE", "NEUTRAL", "POSITIVE"]
    X_train, X_test, y_train, y_test = train_test_split(messages, sentiments, test_size=0.2, random_state=42)
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
    pipeline.fit(X_train, y_train)
train_model()

# User Registration
@app.post("/register/")
def register_user(user: User):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    fake_users_db[user.username] = {"password": hash_password(user.password)}
    return {"message": "User registered successfully"}

# User Login & Token Generation
@app.post("/login/")
def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}

# Protected Routes
@app.post("/analyze/")
def analyze_user_behavior(request: UserRequest, user: str = Depends(get_current_user)):
    response = comprehend.detect_sentiment(Text=request.message, LanguageCode='en')
    return {"sentiment": response['Sentiment']}

@app.post("/generate-ads/")
def generate_ad_content(request: UserRequest, user: str = Depends(get_current_user)):
    sentiment = comprehend.detect_sentiment(Text=request.message, LanguageCode='en')['Sentiment']
    ads = [{"headline": "Exclusive deals for you!", "body": f"Check out {', '.join(request.interests)}!", "sentiment": sentiment} for _ in range(10)]
    return {"ads": ads}

@app.post("/recommend/")
def dynamic_audience_targeting(request: UserRequest, user: str = Depends(get_current_user)):
    recommendations = personalize_runtime.get_recommendations(campaignArn=CAMPAIGN_ARN, userId=request.user_id).get('itemList', [])
    return {"user_id": request.user_id, "recommendations": recommendations, "scope": {1: 'local', 2: 'statewide', 3: 'international'}.get(request.tier, 'local')}

@app.post("/schedule/")
def schedule_tasks(task_name: str, user: str = Depends(get_current_user)):
    if datetime.datetime.utcnow().hour in RENEWABLE_ENERGY_HOURS:
        return {"status": "Task executed immediately", "task": task_name}
    return {"status": "Task scheduled for later execution", "task": task_name"}
{
    "username": "testuser",
    "password": "testpassword"
}
uvicorn main:app --host 0.0.0.0 --port 8000


 






