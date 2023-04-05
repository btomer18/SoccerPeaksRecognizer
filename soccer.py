from typing import List

from pydub import AudioSegment
import numpy as np
from moviepy.editor import *
import wave
import json

from vosk import Model, KaldiRecognizer

SOCCER_WORDS = [
    "advantage",
    "angle",
    "backheel",
    "boot",
    "boundary",
    "box",
    "buildup",
    "bursts",
    "chip",
    "clearance",
    "combo",
    "crossbar",
    "cutback",
    "deflection",
    "dink",
    "direct",
    "disallowed",
    "dive",
    "dummies",
    "dummy",
    "flick",
    "follow-through",
    "footwork",
    "formation",
    "ghost goal",
    "header",
    "hook",
    "indirect",
    "injury",
    "inswinging",
    "juggle",
    "knuckleball",
    "lead",
    "lofted",
    "long ball",
    "marking",
    "nutmeg",
    "offside",
    "one-two",
    "outfield",
    "overhead",
    "panenka",
    "penalty",
    "pitch",
    "possession",
    "precision",
    "pressure",
    "pull-back",
    "punch",
    "quick feet",
    "rabona",
    "rebound",
    "redirect",
    "rhythm",
    "roster",
    "scissor kick",
    "shootout",
    "sliding tackle",
    "spin",
    "spin shot",
    "sprint",
    "stoppage",
    "strike",
    "swing",
    "tap-in",
    "through ball",
    "timewasting",
    "top corner",
    "touchline",
    "trickery",
    "turnover",
    "ultimate",
    "volley",
    "winger",
    "zonal",
    "amazing",
    "awesome",
    "beautiful",
    "breathtaking",
    "brilliant",
    "celebration",
    "climactic",
    "dramatic",
    "electric",
    "epic",
    "exciting",
    "explosive",
    "fantastic",
    "feisty",
    "ferocious",
    "fierce",
    "fiery",
    "flawless",
    "forceful",
    "formidable",
    "frenzied",
    "glorious",
    "gritty",
    "heroic",
    "impressive",
    "incendiary",
    "incredible",
    "intense",
    "jaw-dropping",
    "legendary",
    "magnificent",
    "majestic",
    "marvelous",
    "mesmerizing",
    "mighty",
    "momentous",
    "monumental",
    "outstanding",
    "passionate",
    "penalty",
    "powerful",
    "riveting",
    "scintillating",
    "sensational",
    "spectacular",
    "speedy",
    "splendid",
    "stunning",
    "superb",
    "supreme",
    "surprising",
    "tantalizing",
    "tense",
    "terrific",
    "thrilling",
    "thunderous",
    "top-class",
    "triumphant",
    "unbelievable",
    "unforgettable",
    "unstoppable",
    "vibrant",
    "victorious",
    "vigorous",
    "wonderful",
    "zealous",
    "amazingly",
    "brilliantly",
    "dazzlingly",
    "dramatically",
    "excitingly",
    "exhilaratingly",
    "extraordinarily",
    "fantastically",
    "fiercely",
    "gloriously",
    "gracefully",
    "heroically",
    "impressively",
    "incredibly",
    "intensely",
    "majestically",
    "mind-blowing",
    "phenomenally",
    "remarkably",
    "rivetingly",
    "spectacularly",
    "stupendously",
    "surprisingly",
    "thrillingly",
    "unbelievably",
    "unforgettably",
]


class Word:
    """A class representing a word from the JSON format for vosk speech recognition API"""

    def __init__(self, dict):
        """
        Parameters:
          dict (dict) dictionary from JSON, containing:
            conf (float): degree of confidence, from 0 to 1
            end (float): end time of the pronouncing the word, in seconds
            start (float): start time of the pronouncing the word, in seconds
            word (str): recognized word
        """

        self.conf = dict["conf"]
        self.end = dict["end"]
        self.start = dict["start"]
        self.word = dict["word"]

    def to_string(self):
        """Returns a string describing this instance"""
        return "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
            self.word, self.start, self.end, self.conf * 100
        )


def mp4_to_wav_mono(filename_no_ext):
    # Load the audio file using moviepy
    video = VideoFileClip(f"{filename_no_ext}.mp4")
    audio = video.audio
    audio.write_audiofile(f"{filename_no_ext}.wav", ffmpeg_params=["-ac", "1"])


def audio_to_text(filename_no_ext):

    # Load the audio file using moviepy
    video = VideoFileClip(f"{filename_no_ext}.mp4")
    audio = video.audio
    audio.write_audiofile(f"{filename_no_ext}.wav", ffmpeg_params=["-ac", "1"])

    model_path = "vosk-model-en-us-0.22-lgraph"

    model = Model(model_path)
    wf = wave.open(f"{filename_no_ext}.wav", "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    # get the list of JSON dictionaries
    results = []
    # recognize speech using vosk model
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part_result = json.loads(rec.Result())
            results.append(part_result)
    part_result = json.loads(rec.FinalResult())
    results.append(part_result)

    # convert list of JSON dictionaries to list of 'Word' objects
    list_of_words = []
    for sentence in results:
        if len(sentence) == 1:
            # sometimes there are bugs in recognition
            # and it returns an empty dictionary
            # {'text': ''}
            continue
        for obj in sentence["result"]:
            w = Word(obj)  # create custom Word object
            list_of_words.append(w)  # and add it to list

    wf.close()  # close audiofile

    # output to the screen
    return list_of_words


def audio_peaks(filename_no_ext):
    # Load the audio file
    audio_file = AudioSegment.from_file(f"{filename_no_ext}.mp4", format="mp4")

    audio_file.set_frame_rate(audio_file.frame_rate)
    # Convert the audio to a numpy array
    audio_array = np.array(audio_file.get_array_of_samples())

    # Set a threshold value for high volume peaks
    threshold = 0.7  # Adjust this value to suit your needs

    # Find the indices where the audio signal crosses the threshold
    indices = np.where(audio_array > threshold * audio_array.max())[0]

    # Print the time of each high volume peak moment
    # I had an issue with frame rate is halved, so I doubled it
    times = [index / (audio_file.frame_rate * 2) for index in indices]
    return high_peak_beginning(times)


def high_peak_beginning(times: List[float]):

    if len(times) == 0:
        return []
    filtered = [times[0]]
    # Iterate over the remaining timestamps
    for i in range(1, len(times)):
        # If the time difference between the current timestamp and the previous
        # timestamp is greater than or equal to 1 second, add it to the output list
        if times[i] - times[i - 1] >= 1.5:
            filtered.append(times[i])

    return filtered


def write_gif(from_second: float, to_second: float, filename_no_ext: str):
    # create directory if not exists
    if not os.path.exists("gifs"):
        os.mkdir("gifs")

    # Load video file
    clip = VideoFileClip(f"{filename_no_ext}.mp4")

    # Trim clip from second 1 to 3
    trimmed_clip = clip.subclip(from_second, to_second)

    # Resize clip to 480x360
    resized_clip = trimmed_clip.resize((480, 360))

    frames = []
    for frame in resized_clip.iter_frames():
        frames.append(frame)

    # Save frames as GIF using imageio
    imageio.mimsave(f"gifs/{from_second}.gif", frames, fps=10)

    # Close clip
    clip.close()


def filter_important_words(words: List[Word]) -> List[float]:
    return [word.start for word in words if word.word in SOCCER_WORDS]


def remove_duplicates(peaks: List[float]) -> List[float]:
    peaks = sorted(peaks)
    return [peaks[0]] + [
        peaks[i] for i in range(1, len(peaks)) if peaks[i] - peaks[i - 1] > 3
    ]


def main():
    filename = "soccer_game"
    text = audio_to_text(filename)
    sound_peaks = audio_peaks(filename)
    text_peaks = filter_important_words(text)
    highlights = sound_peaks + text_peaks
    highlights = remove_duplicates(highlights)

    for highlight in highlights:
        if highlight - 3 < 0:
            continue
        write_gif(highlight - 3, highlight + 3, filename)


if __name__ == "__main__":
    main()
