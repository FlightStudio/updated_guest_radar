from flask import Flask, render_template, request, redirect, url_for, Response, stream_with_context
from googleapiclient.discovery import build
from google.oauth2 import service_account
from openai import OpenAI
from flask_caching import Cache
import requests
import re
from collections import Counter
from dotenv import load_dotenv
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import csv
import logging
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy import text, insert
import datetime
from dateutil import parser
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug("Starting application")

# Load environment variables from .env file if it exists (for local development)
load_dotenv()

# Now use environment variables directly
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME = os.getenv('DB_NAME')
CLOUD_SQL_CONNECTION_NAME = os.getenv('CLOUD_SQL_CONNECTION_NAME')
logging.debug(f"GOOGLE_CREDENTIALS environment variable exists: {bool(os.environ.get('GOOGLE_CREDENTIALS'))}")

if os.environ.get('GOOGLE_CREDENTIALS'):
    with open('/app/google-credentials.json', 'w') as f:
        json.dump(json.loads(os.environ.get('GOOGLE_CREDENTIALS')), f)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/google-credentials.json'
    logging.debug("Google credentials file created")
else:
    logging.error("GOOGLE_CREDENTIALS environment variable not found")

google_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
logging.debug(f"GOOGLE_APPLICATION_CREDENTIALS: {google_creds}")

logging.debug(f"YOUTUBE_API_KEY exists: {bool(os.environ.get('YOUTUBE_API_KEY'))}")
logging.debug(f"OPENAI_API_KEY exists: {bool(os.environ.get('OPENAI_API_KEY'))}")
logging.debug(f"DB_USER exists: {bool(os.environ.get('DB_USER'))}")
logging.debug(f"DB_PASS exists: {bool(os.environ.get('DB_PASS'))}")
logging.debug(f"DB_NAME exists: {bool(os.environ.get('DB_NAME'))}")
logging.debug(f"CLOUD_SQL_CONNECTION_NAME exists: {bool(os.environ.get('CLOUD_SQL_CONNECTION_NAME'))}")

# Print values for debugging (remove in production)
print(f"YOUTUBE_API_KEY: {YOUTUBE_API_KEY}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
print(f"DB_USER: {DB_USER}")
print(f"DB_PASS: {DB_PASS}")
print(f"DB_NAME: {DB_NAME}")
print(f"CLOUD_SQL_CONNECTION_NAME: {CLOUD_SQL_CONNECTION_NAME}")

required_env_vars = ['YOUTUBE_API_KEY', 'OPENAI_API_KEY', 'DB_USER', 'DB_PASS', 'DB_NAME', 'CLOUD_SQL_CONNECTION_NAME']
for var in required_env_vars:
    if not os.getenv(var):
        logging.error(f"Missing required environment variable: {var}")

app = Flask(__name__)

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Remove the hardcoded SERVICE_ACCOUNT_FILE
# Instead, use the GOOGLE_APPLICATION_CREDENTIALS environment variable
# when you need to authenticate with Google services

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Enable SQLAlchemy logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Database connection configuration
db_user = os.getenv("DB_USER", "postgres")
db_pass = os.getenv("DB_PASS")
db_name = os.getenv("DB_NAME", "guestdiscovery")
connection_name = os.getenv("CLOUD_SQL_CONNECTION_NAME")

print(f"DB_USER: {db_user}")
print(f"DB_PASS: {db_pass}")
print(f"DB_NAME: {db_name}")
print(f"CLOUD_SQL_CONNECTION_NAME: {connection_name}")

# Initialize Connector object
connector = Connector()

# Function to return the database connection
def getconn():
    conn = connector.connect(
        connection_name,
        "pg8000",
        user=db_user,
        password=db_pass,
        db=db_name
    )
    return conn

# Create SQLAlchemy engine using the connection pool
pool = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=getconn,
)

# Function to create tables if they don't exist
def create_tables():
    with pool.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS channels (
                channel_id VARCHAR(255) PRIMARY KEY,
                title TEXT,
                description TEXT,
                subscriber_count INTEGER,
                view_count INTEGER,
                video_count INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id VARCHAR(255) PRIMARY KEY,
                title TEXT,
                description TEXT,
                url TEXT,
                views INTEGER,
                average_views FLOAT,
                overperformance_percentage FLOAT,
                thumbnail_url TEXT,
                guest_info TEXT,
                published_at TIMESTAMP WITH TIME ZONE,
                subscriber_count INTEGER,
                topic TEXT,
                channel_id VARCHAR(255) REFERENCES channels(channel_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS guest_suitability (
                guest_name TEXT PRIMARY KEY,
                suited BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

# Modify the get_youtube_service function to use the API key
def get_youtube_service():
    # credentials = service_account.Credentials.from_service_account_file(
    #     SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/youtube.readonly']
    # )
    # youtube = build("youtube", "v3", credentials=credentials)
    
    # Use the API key instead
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    return youtube

# Cache the YouTube service to avoid re-initializing it frequently
@cache.cached(timeout=3600, key_prefix='youtube_service')
def cached_youtube_service():
    return get_youtube_service()

def calculate_overperformance(video_views, average_views, threshold):
    return video_views > average_views * (threshold / 100)

def get_channel_stats(youtube, channel_id):
    try:
        channel_request = youtube.channels().list(
            part='statistics',
            id=channel_id
        )
        channel_response = channel_request.execute()
        channel_stats = channel_response['items'][0]['statistics']
        channel_views = int(channel_stats['viewCount'])
        channel_videos = int(channel_stats['videoCount'])
        subscriber_count = int(channel_stats.get('subscriberCount', 0))
        return channel_views, channel_videos, subscriber_count
    except Exception as e:
        logging.error(f"Error fetching channel stats: {str(e)}")
        return 0, 0, 0

def extract_guest_info_using_gpt(video_title, video_description):
    # Truncate the description to approximately 100 words
    truncated_description = ' '.join(video_description.split()[:50])
    
    prompt = f"""
    Extract the name of the guest or speaker from the following YouTube video title and description. 
    Provide the result in the following JSON format:
    {{
        "guest_name": "Name Here" 
    }}
    If no guest is identified, set "guest_name" to "Unknown".
    
    Title: {video_title}
    Description: {truncated_description}
    
    Guest Name:
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts guest names from video details."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.3
        )
        guest_info_text = response.choices[0].message.content.strip()
        logging.info(f"Video Title: {video_title}")
        logging.info(f"OpenAI API raw response: {guest_info_text}")
        
        # Try to parse as JSON, if it fails, use the raw text
        try:
            guest_info_json = json.loads(guest_info_text)
            guest_info = guest_info_json.get("guest_name", "Unknown")
        except json.JSONDecodeError:
            # If it's not valid JSON, use the raw text if it's not empty
            guest_info = guest_info_text if guest_info_text else "Unknown"
        
        # Validate and clean the extracted name
        guest_info = clean_guest_name(guest_info)
               # Validate the extracted name
        if not guest_info or guest_info.lower() == "unknown" or "unknown" in guest_info.lower():
            return "Unknown"
        
        logging.info(f"OpenAI API name response: {guest_info}")

        return guest_info
    except Exception as e:
        logging.error(f"Error extracting guest info: {str(e)}")
        return "Unknown"

def clean_guest_name(name):
    # Remove any non-alphabetic characters at the start or end of the name
    name = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', name)
    
    # If the name is empty or "Unknown" after cleaning, return "Unknown"
    if not name or name.lower() == "unknown":
        return "Unknown"
    
    return name

def clean_and_split_guest_names(guest_info):
    # Remove any JSON-like structures
    guest_info = re.sub(r'\{.*?\}', '', guest_info)
    guest_info = re.sub(r'json', '', guest_info, flags=re.IGNORECASE)
    
    # Split the guest info by common separators
    names = re.split(r',|\band\b|\&', guest_info)
    # Clean each name
    cleaned_names = [clean_guest_name(name.strip()) for name in names]
    # Remove any empty strings or 'Unknown'
    return [name for name in cleaned_names if name and name.lower() != 'unknown']

def get_published_after_date(date_filter):
    now = datetime.datetime.now(datetime.timezone.utc)
    if date_filter == 'week':
        published_after = now - datetime.timedelta(weeks=1)
    elif date_filter == 'month':
        published_after = now - datetime.timedelta(days=30)
    elif date_filter == 'year':
        published_after = now - datetime.timedelta(days=365)
    else:
        return None
    
    # Format the date correctly for the YouTube API
    return published_after.strftime('%Y-%m-%dT%H:%M:%SZ')

@cache.memoize(timeout=3600)
def get_trending_videos_total_views(youtube):
    try:
        request = youtube.videos().list(
            part='statistics',
            chart='mostPopular',
            maxResults=50,
            regionCode='US'  # or any other region
        )
        response = request.execute()
        total_trending_views = sum(int(item['statistics']['viewCount']) for item in response['items'])
        return total_trending_views
    except Exception as e:
        logging.error(f"Error fetching trending videos: {str(e)}")
        return 0

def get_guest_suitability():
    suitability = {}
    try:
        with pool.connect() as conn:
            result = conn.execute(text("SELECT guest_name, suited FROM guest_suitability"))
            for row in result:
                suitability[row[0]] = row[1]
    except Exception as e:
        logging.error(f"Error retrieving guest suitability: {str(e)}")
    return suitability

client = OpenAI(api_key=OPENAI_API_KEY)

def enhance_youtube_query(query):
    logging.info(f"Original YouTube Query: {query}")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a query enhancer for YouTube searches about podcasts and interviews. Expand the given query to include 2-3 related terms or synonyms. Focus on creating a short, targeted query that captures the user's intent. Use quotation marks for exact phrases if necessary. Keep the enhanced query concise, under 10 words, and relevant to podcasts and interviews."},
            {"role": "user", "content": f"Enhance this YouTube search query for better podcast and interview results (keep it under 10 words): {query}"}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5
    )
    enhanced_query = response.choices[0].message.content.strip()
    
    # Remove any quotation marks around the entire query, but keep internal quotes
    enhanced_query = re.sub(r'^"|"$', '', enhanced_query)
    
    # Ensure the query doesn't exceed a certain length
    max_query_length = 100
    if len(enhanced_query) > max_query_length:
        enhanced_query = enhanced_query[:max_query_length].rsplit(' ', 1)[0]
    
    logging.info(f"Enhanced YouTube Query: {enhanced_query}")
    return enhanced_query

# Modify the get_youtube_videos function to use the enhanced query
@cache.memoize(timeout=300)
def get_youtube_videos(topic, date_filter, sort_filter, overperformance_threshold):
    logging.info(f"Starting get_youtube_videos with topic: {topic}, date_filter: {date_filter}, sort_filter: {sort_filter}, overperformance_threshold: {overperformance_threshold}")
    youtube = cached_youtube_service()
    videos = []
    channel_data = []
    video_data = []
    video_ids = set()
    guest_counter = Counter()
    total_views = 0
    keywords = [""]
    published_after = get_published_after_date(date_filter)
    guest_suitability = get_guest_suitability()

    # Enhance the topic query
    enhanced_topic = enhance_youtube_query(topic)

    # Fetch fresh data from YouTube API
    for i, keyword in enumerate(keywords):
        query = f'{enhanced_topic} {keyword}'
        try:
            logging.info(f"Searching YouTube for query: {query}")
            published_after_iso = get_published_after_date(date_filter)
            request = youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=50,
                publishedAfter=published_after_iso,
                videoDuration='long',
                order='relevance',
                regionCode='US',  # Add this line to specify the region
                relevanceLanguage='en'  # Add this line to prefer English content
            )
            response = request.execute()
            logging.info(f"Found {len(response['items'])} videos from YouTube search")
            
            yield f"data: Processed {i+1}/{len(keywords)} keywords\n\n"
            
            for item in response['items']:
                try:
                    video_id = item['id']['videoId']
                    if video_id in video_ids:
                        logging.info(f"Skipping duplicate video ID: {video_id}")
                        continue  # Skip duplicate videos
                    video_ids.add(video_id)
                    
                    video_data_item = process_video_item(item, youtube, topic)
                    videos.append(video_data_item)
                    total_views += video_data_item['views']
                    guest_names = clean_and_split_guest_names(video_data_item['guest_info'])
                    guest_counter.update(guest_names)

                    # Collect data for bulk insert
                    channel_data.append({
                        "channel_id": video_data_item['channel_id'],
                        "title": video_data_item['title'],
                        "subscriber_count": video_data_item['subscriber_count'],
                        "view_count": video_data_item['views'],
                        "video_count": 1  # This is still a placeholder
                    })

                    video_data.append({
                        "video_id": video_data_item['url'].split('=')[1],
                        "title": video_data_item['title'],
                        "description": video_data_item['description'],
                        "url": video_data_item['url'],
                        "views": video_data_item['views'],
                        "average_views": video_data_item['average_views'],
                        "overperformance_percentage": video_data_item['overperformance_percentage'],
                        "thumbnail_url": video_data_item['thumbnail_url'],
                        "guest_info": video_data_item['guest_info'],
                        "published_at": video_data_item['published_at'],
                        "subscriber_count": video_data_item['subscriber_count'],
                        "topic": video_data_item['topic'],
                        "channel_id": video_data_item['channel_id']
                    })

                except Exception as e:
                    logging.error(f"Error processing video: {str(e)}")
                    continue  # Skip to the next video instead of stopping
        except Exception as e:
            logging.error(f"Error fetching videos from YouTube: {str(e)}")

    # Perform bulk insert
    update_database_with_bulk_data(channel_data, video_data)

    # Fetch additional data from database
    db_videos = fetch_additional_videos_from_db(topic, date_filter)
    for video in db_videos:
        if video['url'].split('=')[1] not in video_ids:  # Check for duplicates
            videos.append(video)
            video_ids.add(video['url'].split('=')[1])
            total_views += video['views']
            guest_names = clean_and_split_guest_names(video['guest_info'])
            guest_counter.update(guest_names)

    # Calculate popularity score
    total_trending_views = get_trending_videos_total_views(youtube)
    popularity_score = (total_views / total_trending_views) * 100 if total_trending_views > 0 else 0
    logging.info(f"Popularity score for topic '{topic}': {popularity_score}")

    # Apply filters and sort
    filtered_videos = filter_and_sort_videos(videos, guest_suitability, overperformance_threshold, sort_filter)

    # Get the top 5 guests based on appearance count
    top_guests = guest_counter.most_common(5)
    logging.info(f"Top guests: {top_guests}")
    logging.info(f"Returning {len(filtered_videos)} filtered videos")

    return filtered_videos[:50], total_views, popularity_score, top_guests

def process_video_item(item, youtube, topic):
    video_id = item['id']['videoId']
    channel_id = item['snippet']['channelId']
    video_title = item['snippet']['title']
    video_description = item['snippet']['description']
    thumbnail_url = item['snippet']['thumbnails']['default']['url']

    # Fetch video statistics
    video_request = youtube.videos().list(
        part='statistics,snippet',
        id=video_id
    )
    video_response = video_request.execute()
    video_stats = video_response['items'][0]['statistics']
    video_views = int(video_stats['viewCount'])
    published_at_str = video_response['items'][0].get('snippet', {}).get('publishedAt', '')
    published_at = parser.isoparse(published_at_str) if published_at_str else None

    # Fetch channel statistics
    channel_views, channel_videos, subscriber_count = get_channel_stats(youtube, channel_id)
    average_views = channel_views / channel_videos if channel_videos > 0 else 0

    # Calculate overperformance percentage
    overperformance_percentage = ((video_views - average_views) / average_views) * 100 if average_views > 0 else 0

    # Extract guest info
    guest_info = extract_guest_info_using_gpt(video_title, video_description)

    return {
        'title': video_title,
        'description': video_description,
        'url': f"https://www.youtube.com/watch?v={video_id}",
        'views': video_views,
        'average_views': average_views,
        'overperformance_percentage': overperformance_percentage,
        'thumbnail_url': thumbnail_url,
        'guest_info': guest_info,
        'published_at': published_at,
        'subscriber_count': subscriber_count,
        'topic': topic,
        'channel_id': channel_id
    }

def fetch_additional_videos_from_db(topic, date_filter):
    with pool.connect() as conn:
        published_after = get_published_after_date(date_filter)
        query = text("""
            SELECT title, description, url, views, average_views, overperformance_percentage,
                   thumbnail_url, guest_info, published_at, subscriber_count, topic, channel_id
            FROM videos 
            WHERE topic = :topic
            AND published_at > :published_after
            ORDER BY published_at DESC
        """)
        result = conn.execute(query, {"topic": topic, "published_after": published_after})
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result]

def filter_and_sort_videos(videos, guest_suitability, overperformance_threshold, sort_filter):
    filtered_videos = []
    for video in videos:
        if calculate_overperformance(video['views'], video['average_views'], overperformance_threshold):
            guest_name = video['guest_info'].split(',')[0].strip()
            if guest_suitability.get(guest_name, True):
                filtered_videos.append(video)
                logging.info(f"Video '{video['title']}' overperforms. Including in results.")
            else:
                logging.info(f"Guest '{guest_name}' marked as unsuitable. Excluding video '{video['title']}'.")
        else:
            logging.info(f"Video '{video['title']}' does not overperform. Skipping.")

    if sort_filter == 'views':
        filtered_videos.sort(key=lambda x: x['views'], reverse=True)
    elif sort_filter == 'overperformance':
        filtered_videos.sort(key=lambda x: x['overperformance_percentage'], reverse=True)
    elif sort_filter == 'newest':
        filtered_videos.sort(key=lambda x: x['published_at'] or datetime.datetime.min, reverse=True)

    return filtered_videos

def get_mock_data():
    guest_suitability = get_guest_suitability()
    mock_videos = [
        {
            'title': 'Mock Video with John Doe',
            'description': 'This is a mock video description',
            'url': 'https://www.youtube.com/watch?v=mock1',
            'views': 10000,
            'average_views': 5000,
            'overperformance_percentage': 100,
            'thumbnail_url': 'https://via.placeholder.com/120x90.png',
            'guest_info': 'John Doe, Expert',
            'published_at': parser.isoparse('2023-01-01T00:00:00Z'),
            'subscriber_count': 100000
        },
        {
            'title': 'Mock Video with Jane Smith',
            'description': 'Another mock video description',
            'url': 'https://www.youtube.com/watch?v=mock2',
            'views': 15000,
            'average_views': 7000,
            'overperformance_percentage': 114,
            'thumbnail_url': 'https://via.placeholder.com/120x90.png',
            'guest_info': 'Jane Smith, Researcher',
            'published_at': parser.isoparse('2023-02-01T00:00:00Z'),
            'subscriber_count': 120000
        },
    ]

    filtered_videos = [video for video in mock_videos if guest_suitability.get(video['guest_info'].split(',')[0].strip(), True)]

    return {
        'videos': filtered_videos,
        'total_views': sum(video['views'] for video in filtered_videos),
        'popularity_score': 75.5,
        'top_guests': [
            (video['guest_info'], 1) for video in filtered_videos[:5]
        ]
    }

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def index():
    if app.config.get('DEBUG', False):
        logging.info("Running in DEBUG mode, using mock data")
        mock_data = get_mock_data()
        return render_template('index.html', videos=mock_data['videos'], total_views=mock_data['total_views'], 
                               popularity_score=mock_data['popularity_score'], top_guests=mock_data['top_guests'])
    else:
        logging.info("Running in production mode")
        if request.method == 'POST':
            topic = request.form.get('topic', '')
            date_filter = request.form.get('date_filter', 'any')
            sort_filter = request.form.get('sort_filter', 'views')
            overperformance_threshold = int(request.form.get('overperformance_threshold', 120))
            
            if topic:
                return Response(stream_with_context(stream_results(topic, date_filter, sort_filter, overperformance_threshold)), content_type='text/event-stream')
        
        return render_template('index.html', videos=[], total_views=0, popularity_score=0, top_guests=[])

def stream_results(topic, date_filter, sort_filter, overperformance_threshold):
    yield "data: Processing request...\n\n"
    
    videos, total_views, popularity_score, top_guests = get_youtube_videos(topic, date_filter, sort_filter, overperformance_threshold)
    
    yield f"data: {json.dumps({'videos': videos, 'total_views': total_views, 'popularity_score': popularity_score, 'top_guests': top_guests})}\n\n"
    
    yield "data: Done\n\n"

@app.route('/update_guest_suitability', methods=['POST'])
@limiter.limit("5 per minute")
def update_guest_suitability():
    guest_name = request.form.get('guest_name')
    suitability = request.form.get('suitability')
    suited = True if suitability == 'suited' else False

    logging.info(f"Updating guest: {guest_name}, suitability: {suited}")

    try:
        with pool.connect() as conn:
            try:
                with conn.begin():
                    conn.execute(
                        text("INSERT INTO guest_suitability (guest_name, suited) VALUES (:guest_name, :suited) ON CONFLICT (guest_name) DO UPDATE SET suited = :suited"),
                        {"guest_name": guest_name, "suited": suited}
                    )
            except Exception as e:
                logging.error(f"Error inserting/updating guest suitability: {str(e)}")
                return "Error", 500
        logging.info("Database updated successfully")

        # Clear the cache for get_youtube_videos
        cache.delete_memoized(get_youtube_videos)

        return "Success", 200
    except Exception as e:
        logging.error(f"Error updating database: {str(e)}")
        return "Error", 500

def update_database_with_bulk_data(channel_data, video_data):
    with pool.connect() as conn:
        try:
            with conn.begin():
                # Bulk insert/update channels
                conn.execute(
                    insert(text('channels')).values(channel_data).on_conflict_do_update(
                        index_elements=['channel_id'],
                        set_={
                            'title': text('EXCLUDED.title'),
                            'subscriber_count': text('EXCLUDED.subscriber_count'),
                            'view_count': text('EXCLUDED.view_count'),
                            'video_count': text('EXCLUDED.video_count'),
                            'last_updated': text('CURRENT_TIMESTAMP')
                        }
                    )
                )

                # Bulk insert/update videos
                conn.execute(
                    insert(text('videos')).values(video_data).on_conflict_do_update(
                        index_elements=['video_id'],
                        set_={
                            'views': text('EXCLUDED.views'),
                            'average_views': text('EXCLUDED.average_views'),
                            'overperformance_percentage': text('EXCLUDED.overperformance_percentage'),
                            'guest_info': text('EXCLUDED.guest_info'),
                            'subscriber_count': text('EXCLUDED.subscriber_count'),
                            'published_at': text('EXCLUDED.published_at'),
                            'topic': text('EXCLUDED.topic')
                        }
                    )
                )
        except Exception as e:
            logging.error(f"Error updating database in bulk: {str(e)}")

if __name__ == '__main__':
    logging.info("Starting the application")
    create_tables()
    app.run(debug=False)

def update_database_with_bulk_data(channel_data, video_data):
    with pool.connect() as conn:
        try:
            with conn.begin():
                # Bulk insert/update channels
                conn.execute(
                    insert(text('channels')).values(channel_data).on_conflict_do_update(
                        index_elements=['channel_id'],
                        set_={
                            'title': text('EXCLUDED.title'),
                            'subscriber_count': text('EXCLUDED.subscriber_count'),
                            'view_count': text('EXCLUDED.view_count'),
                            'video_count': text('EXCLUDED.video_count'),
                            'last_updated': text('CURRENT_TIMESTAMP')
                        }
                    )
                )

                # Bulk insert/update videos
                conn.execute(
                    insert(text('videos')).values(video_data).on_conflict_do_update(
                        index_elements=['video_id'],
                        set_={
                            'views': text('EXCLUDED.views'),
                            'average_views': text('EXCLUDED.average_views'),
                            'overperformance_percentage': text('EXCLUDED.overperformance_percentage'),
                            'guest_info': text('EXCLUDED.guest_info'),
                            'subscriber_count': text('EXCLUDED.subscriber_count'),
                            'published_at': text('EXCLUDED.published_at'),
                            'topic': text('EXCLUDED.topic')
                        }
                    )
                )
        except Exception as e:
            logging.error(f"Error updating database in bulk: {str(e)}")

if __name__ == '__main__':
    logging.info("Starting the application")
    create_tables()
    app.run(debug=False)