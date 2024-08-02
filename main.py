import argparse
import os
import whisperx

from datetime import timedelta
from dotenv import dotenv_values

config = dotenv_values(".env")


video_extensions = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".3gp",
}


def transcribe_video(videoPath: str) -> list:
    model = whisperx.load_model(
        config["MODEL"], device=config["DEVICE"], compute_type=config["COMPUTE_TYPE"]
    )
    audio = whisperx.load_audio(videoPath)

    transcribedText = model.transcribe(
        audio, batch_size=int(config["BATCH_SIZE"]), language=config["LANGUAGE"]
    )

    # sync subtitles
    modelAlign, metadata = whisperx.load_align_model(
        language_code=transcribedText["language"], device=config["DEVICE"]
    )
    result = whisperx.align(
        transcribedText["segments"],
        modelAlign,
        metadata,
        audio,
        config["DEVICE"],
        return_char_alignments=False,
    )

    return result["segments"]


def create_subtitles(videoPath: str, segments: list):
    subtitlesName = change_extension_to_srt(videoPath)

    # if srt file exists, delete it
    if os.path.exists(subtitlesName):
        os.remove(subtitlesName)

    for index, segment in enumerate(segments):
        startTime = str(0) + str(timedelta(seconds=int(segment["start"]))) + ",000"
        endTime = str(0) + str(timedelta(seconds=int(segment["end"]))) + ",300"
        text = segment["text"].strip()

        subtitle = f"{index + 1}\n{startTime} --> {endTime}\n{text}\n\n"

        with open(subtitlesName, "a", encoding="utf-8") as srtFile:
            srtFile.write(subtitle)

    print(f"\nSubtitles has been created: '{subtitlesName}'\n")


def change_extension_to_srt(videoPath):
    base = os.path.splitext(videoPath)[0]
    newPath = base + ".srt"

    return os.path.join(newPath)


def is_video_file(path: str):
    _, ext = os.path.splitext(path)

    return ext.lower() in video_extensions


def main(path: str):
    if os.path.isfile(path) and is_video_file(path):
        segments = transcribe_video(path)
        create_subtitles(path, segments)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            videoPath = os.path.join(path, filename)
            if os.path.isfile(videoPath) and is_video_file(videoPath):
                segments = transcribe_video(videoPath)
                create_subtitles(videoPath, segments)
    else:
        print(f"'{path}' is not a video file or a directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="subtitels creater")
    parser.add_argument("-p", "--path", type=str, required=True, help="video path")

    args = parser.parse_args()
    main(args.path)
