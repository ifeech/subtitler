import argparse
import os
import whisperx

from datetime import timedelta
from dotenv import dotenv_values

config = dotenv_values(".env")


def transcribe_video(videoPath):
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


def create_subtitles(subtitlesName, segments):
    subtitlesName = os.path.join(subtitlesName)

    # if srt file exists, delete it
    if os.path.exists(subtitlesName):
        os.remove(subtitlesName)
    for index, segment in enumerate(segments):
        startTime = str(0) + str(timedelta(seconds=int(segment["start"]))) + ",000"
        endTime = str(0) + str(timedelta(seconds=int(segment["end"]))) + ",700"
        text = segment["text"].strip()

        subtitle = f"{index + 1}\n{startTime} --> {endTime}\n{text}\n\n"

        with open(subtitlesName, "a", encoding="utf-8") as srtFile:
            srtFile.write(subtitle)

    return subtitlesName


def main(videoPath):
    segments = transcribe_video(videoPath)

    subtitlesName = videoPath[:-4] + "_subtitles.srt"

    print(
        f"***Subtitles has been created: {create_subtitles(subtitlesName, segments)}***"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="subtitels creater")
    parser.add_argument("-p", "--path", type=str, required=True, help="video path")

    args = parser.parse_args()
    main(args.path)
