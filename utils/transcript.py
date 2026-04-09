from youtube_transcript_api import YouTubeTranscriptApi
import http.cookiejar
import requests
import re
import os


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def _make_api() -> YouTubeTranscriptApi:
    """Build YouTubeTranscriptApi, injecting cookies.txt if it exists."""
    cookies_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "cookies.txt")
    )
    if os.path.exists(cookies_path):
        jar = http.cookiejar.MozillaCookieJar()
        jar.load(cookies_path, ignore_discard=True, ignore_expires=True)
        session = requests.Session()
        session.cookies = jar
        return YouTubeTranscriptApi(http_client=session)
    return YouTubeTranscriptApi()


def get_transcript(video_id: str) -> str:
    """Fetch transcript for a YouTube video and return as plain text."""
    api = _make_api()
    transcript = api.fetch(video_id)
    return " ".join(entry.text for entry in transcript)
