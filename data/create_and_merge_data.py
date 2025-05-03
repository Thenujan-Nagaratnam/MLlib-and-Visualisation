import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import re
import os


# load .env file
from dotenv import load_dotenv

# Spotify API credentials
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
client_credentials_manager = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET
)
# Initialize the Spotify API client
sp = spotipy.Spotify(
    client_credentials_manager=client_credentials_manager, requests_timeout=40
)  # spotify object to access API

# Create a Genius API client
GENIUS_CLIENT_ACCESS_TOKEN = (
    "UKI6erErQ37IxWsTlepJu4cffy4Que7gvV1r9ydY8nVLeRMlulgCPhFld9PaMhD5"
)
genius = lyricsgenius.Genius(GENIUS_CLIENT_ACCESS_TOKEN, timeout=40)


def lyrics(song, artist):
    genius = lyricsgenius.Genius("")
    song = genius.search_song(title=song, artist=artist)
    try:
        lyrics = song.lyrics
        print(lyrics)
        print("\n")
    except AttributeError:
        print("There are no lyrics for this song on Genius")
        print("\n")


def scrape_lyrics(artistname, songname):
    artistname2 = (
        str(artistname.replace(" ", "-")) if " " in artistname else str(artistname)
    )
    songname2 = str(songname.replace(" ", "-")) if " " in songname else str(songname)
    page = requests.get(
        "https://genius.com/" + artistname2 + "-" + songname2 + "-" + "lyrics"
    )
    html = BeautifulSoup(page.text, "html.parser")
    lyrics1 = html.find("div", class_="lyrics")
    lyrics2 = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-2 jgQsqn")
    if lyrics1:
        lyrics = lyrics1.get_text()
    elif lyrics2:
        lyrics = lyrics2.get_text()
    elif lyrics1 == lyrics2 == None:
        lyrics = None
    return lyrics


def lyrics_onto_frame(df1, artist_name):
    for i, x in enumerate(df1["Track Name"]):
        test = scrape_lyrics(artist_name, x)
        print(test)
        df1.loc[i, "Lyrics"] = test
    return df1


artist_names = []
track_names = []
release_dates = []
genres = []
lyrics = []

i = 1

# Collect up to 200 songs (50 per request, offset in steps of 50)
for offset in range(0, 200, 50):
    pop_tracks = sp.search(q="genre:hyperpop", type="track", limit=50, offset=offset)

    for track in pop_tracks["tracks"]["items"]:
        try:
            print(i)
            i += 1

            song = genius.search_song(track["name"], track["artists"][0]["name"])
            if not song:
                continue
            song = song.to_dict()
            l = song["lyrics"]
            l = re.sub(r"^.*?\n+", "", l, count=1)
            l = re.sub(r"\s+", " ", l)

            lyrics.append(l)
            artist_names.append(track["artists"][0]["name"])
            track_names.append(track["name"])
            release_dates.append(track["album"]["release_date"])
            genres.append("hyperpop")

        except Exception as e:
            print(f"Error processing song: {e}")
            continue

# Create DataFrame
data = {
    "artist_name": artist_names,
    "track_name": track_names,
    "release_date": release_dates,
    "genre": genres,
    "lyrics": lyrics,
}

df = pd.DataFrame(data)

# Convert and clean date
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df = df.dropna(subset=["release_date"])
df["release_date"] = df["release_date"].dt.year.astype("int64")

# # Save to CSV
# df.to_csv("hyperpop.csv", index=False)


###############################################################################################################################
###############################################################################################################################


def remove_non_ascii(text):
    return re.sub(r"[^\x00-\x7F]+", "", text)


# # Load and clean the data
# df = pd.read_csv("hyperpop.csv")
df = df.dropna(subset=["lyrics"])
df = df.drop_duplicates(subset=["lyrics"])
df = df.drop_duplicates(subset=["track_name"])

# Remove non-ASCII characters from lyrics
df["lyrics"] = df["lyrics"].apply(remove_non_ascii)
df["lyrics"] = df["lyrics"].str.replace(r"\[.*?\]", "", regex=True)

# # Save cleaned data
df.to_csv("Student_dataset.csv", index=False)


###############################################################################################################################
###############################################################################################################################

mendeley_df = pd.read_csv("mendeley-music-dataset.csv")
# student_df = pd.read_csv("Student_dataset.csv")

student_df = df

mendeley_df = mendeley_df[
    ["artist_name", "track_name", "release_date", "genre", "lyrics"]
]
merged_dataset = pd.concat([mendeley_df, student_df], ignore_index=True)

merged_dataset.to_csv("Merged_dataset.csv", index=False)
