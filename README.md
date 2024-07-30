# Creator of subtitles for videos

> Tested on python 3.11

Under the hood [whisperx](https://github.com/m-bain/whisperX)

## Function

- Create .srt file based on video
- Support various languages

## Installation

1. Clone a repository
2. Create a virtual environment
```python -m venv .venv```
3. Activating the virtual environment
**Windows**: ```.venv\Scripts\activate```
**macOS и Linux**: ```source .venv/bin/activate```
4. Dependency installation
```pip install -r requirements.txt```
> It is recommended to use cuda.
> If cuda is not supported exclude torchaudio, torchvision, torchvision from dependencies.

## Usage

**.env.example** -> **.env** and specify values for environment variables.

```python main.py --path <path_to_video_file>```