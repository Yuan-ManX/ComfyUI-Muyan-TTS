import os
from torchaudio.datasets import LIBRISPEECH
from sovits.utils import clean_path
from sovits.text.cleaner import clean_text

def process_transcription(text):
    """Process transcription: convert to lowercase, capitalize first letter, add period at the end"""
    # Convert to lowercase
    text = text.lower()
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    # Add period if not present
    if text and not text.endswith('.'):
        text += '.'
    return text


def generate_raw_data_list(librispeech_dir, output_dir="data", subset="dev-clean"):
    """
    generate name2text.txt 
    example:
    librispeech_dir = "path_to_LibriSpeech"  # the path of librispeech
    output_dir = "data"  # output filename
    subset = "dev-clean"       # librispeech subset
    """
    dataset = LIBRISPEECH(root=librispeech_dir, url=subset, download=False)
    os.makedirs(os.path.join(output_dir, "tmp" ), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tmp", "text" ), exist_ok=True)
    specified_speaker = "3752"
    with open(os.path.join(output_dir, "tmp", "text", "name2text.txt"), "w") as f:
        for idx, (waveform, sample_rate, transcription, speaker_id, chapter_id, utterance_id) in enumerate(dataset):
            # Construct audio file path (original .flac)
            if specified_speaker == None:
                # choose the first speaker as example
                specified_speaker = str(speaker_id)
            if str(speaker_id) != specified_speaker:
                continue
            audio_path = os.path.join(librispeech_dir, subset, str(speaker_id), str(chapter_id), f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac")
            audio_path = os.path.basename(clean_path(audio_path))
            _, _, norm_text = clean_text(
                        process_transcription(transcription).replace("%", "-").replace("￥", ","), "en", "v2"
                    )
            
            # Write to name2text.txt
            line = f"{audio_path}\t{norm_text}\n"
            f.write(line)
            
    with open(os.path.join(output_dir, "tmp", "text", "raw_data.list"), "w") as f:
        for idx, (waveform, sample_rate, transcription, speaker_id, chapter_id, utterance_id) in enumerate(dataset):
            # Construct audio file path (original .flac)
            if str(speaker_id) != specified_speaker:
                continue
            
            audio_path = os.path.join(librispeech_dir, "LibriSpeech", subset, str(speaker_id), str(chapter_id), f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac")
            
            # Verify audio file exists
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file {audio_path} not found.")
                continue
            
            # Process transcription
            processed_transcription = process_transcription(transcription)
            
            # Write to raw_data.list
            line = f"{audio_path}|{speaker_id}|EN|{processed_transcription}\n"
            f.write(line)
