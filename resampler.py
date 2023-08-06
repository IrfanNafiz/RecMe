import os


def resample(path_of_files_to_be_resampled):
    command = (
        "for /r %dir in (" + path_of_files_to_be_resampled + ") do ("
        "for %file in (%dir\\*.wav) do ("
        "ffprobe -hide_banner -loglevel panic -show_streams "
        "%file | findstr sample_rate | for /f %%i in ('findstr /r \"[0-9]*\"') do ("
        "set sample_rate=%%i"
        ")"
        "if !sample_rate! neq 16000 ("
        "ffmpeg -hide_banner -loglevel panic -y "
        "-i %file -ar 16000 temp.wav"
        "move /y temp.wav %file"
        ")"
        ")"
    )
    os.system(command)
