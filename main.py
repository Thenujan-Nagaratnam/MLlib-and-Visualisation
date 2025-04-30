import spotipy
import sys, time
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import csv


genius = lyricsgenius.Genius(
    "UKI6erErQ37IxWsTlepJu4cffy4Que7gvV1r9ydY8nVLeRMlulgCPhFld9PaMhD5"
)

"""
download_playlist_tracks.py
    Searches for playlists according to a given string and saves it to a file

    Created in 21/07/2017
    by Vinicius Moura Longaray 
"""


# ############### BEGIN CLASS ############### #
class SpotifyPlaylist:
    seq = 0
    objects = []

    def __init__(self, index, playlistName, playlistId, playlistUsername, playlistSize):
        self.index = index
        self.playlistName = playlistName
        self.playlistId = playlistId
        self.playlistUsername = playlistUsername
        self.playlistSize = playlistSize

        self.__class__.seq += 1
        self.id = self.__class__.seq
        self.__class__.objects.append(self)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<{}: {} - {} - Size:{} - {} - {}>\n".format(
            self.__class__.__name__,
            self.index,
            self.playlistName,
            self.playlistSize,
            self.playlistUsername,
            self.playlistId,
        )

    # < necessary to iterate through object
    def __iter__(self):
        return iter(self.objects)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, item):
        return self.objects[item]

    # necessary to iterate through object />

    @classmethod
    def reset(cls):
        cls.objects = []

    @classmethod
    def all(cls):
        return cls.objects


# ############### END CLASS ############### #


def log(str):
    sys.stdout.buffer.write(
        "[{}] {}\n".format(time.strftime("%H:%M:%S"), str).encode(
            sys.stdout.encoding, errors="replace"
        )
    )
    sys.stdout.flush()


def show_tracks(sp, tracks, writer):

    if "items" in tracks and tracks["items"]:
        for i, item in enumerate(tracks["items"]):
            track = item["track"]
            artist_name = track["artists"][0]["name"]
            track_name = track["name"]
            track_id = track["id"]

            # Get album release date
            release_date = track["album"]["release_date"][:4]

            # Get genres from the artist
            artist_id = track["artists"][0]["id"]
            artist_info = sp.artist(artist_id)
            genres = artist_info.get("genres", [])

            # print(f"Fetching lyrics for {track_name} by {artist_name}...")
            lyrics = get_lyrics(artist_name, track_name, track_id)
            # print(lyrics)

            writer.writerow(
                [
                    track_id,
                    artist_name,
                    track_name,
                    release_date,
                    " ".join(map(str, genres)),
                    lyrics,
                ]
            )

            print(
                f"{i+1:02d}. Artist: {artist_name}, Track: {track_name}, Release: {release_date}, Genre: {genres}"
            )


def get_lyrics(artist_name, track_name, track_id):
    try:
        song = genius.search_song(title=track_name, artist=artist_name)
        return song.lyrics if song else "Lyrics not found"
    except Exception as e:
        print(f"Error fetching lyrics for {track_name} by {artist_name}: {e}")
        return "Lyrics not found"


def main():
    SpotifyPlaylist.reset()

    # Client Credentials Flow
    client_credentials_manager = SpotifyClientCredentials(
        "1e0976a544924864ba6e9390f6e69993", "671c4ed6238b4c8cb7fcfa10db3ccc3a"
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    log("testando isso aqui")
    # Input data
    string = input("Enter a playlists' name you want to find: ")
    # string = 'Indie'

    # Search for the given string
    results = sp.search(string, type="playlist")
    print()
    print(results["playlists"]["items"])

    # Show the results and save it to an internal structure
    for i, t in enumerate(results["playlists"]["items"]):
        if t:
            print(
                " %d %32.32s - Total tracks: %d" % (i, t["name"], t["tracks"]["total"])
            )
            SpotifyPlaylist(
                i, t["name"], t["id"], t["owner"]["id"], t["tracks"]["total"]
            )

    """print()
    print(SpotifyPlaylist.all())"""

    # Choose the option you want to download
    strOption = input("\nEnter the option you want it: ")
    option = int(strOption)
    while option > i:
        log("Invalid option")
        strOption = input("Enter the option you want it: ")
        option = int(strOption)

    # option = 0

    # Get all the tracks inside that playlist
    print(
        "\n################################# TRACKS #################################\n"
    )
    results = sp.user_playlist_tracks(
        SpotifyPlaylist.objects[option].playlistUsername,
        SpotifyPlaylist.objects[option].playlistId,
        fields="items, next",
        limit=100,  # Max allowed per call
    )

    tracks = results

    # Loop through all pages
    with open("playlist_tracks.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Index", "artist_name", "track_name", "release_date", "genre", "lyrics"]
        )

        while tracks:
            show_tracks(sp, tracks, writer)
            if tracks["next"]:
                tracks = sp.next(tracks)
            else:
                break


if __name__ == "__main__":
    main()
