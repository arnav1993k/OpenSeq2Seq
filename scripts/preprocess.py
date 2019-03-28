import os
import librosa
import soundfile as sf
import csv
import re
from . import text_cleaner
extensions = ("wav", "flac", "mp3")

def speech_to_wav(inp_file,outdirname, freq=16000):

  if not os.path.isdir(outdirname):
    os.makedirs(outdirname)

  filename = inp_file.split("/")[-1].split(".")[0]
  wave, sr = librosa.load(inp_file, sr=freq)
  outname = os.path.join(outdirname, filename + ".wav")
  sf.write(outname, wave, sr)
  return outname


def dir_to_wav(dirname, freq=16000, csv_out=False, csv_fname="sample.csv"):
  files = os.listdir(dirname)
  output_rows= []

  for f in files:
    ext = f.split(".")[-1]
    if ext in extensions:
      wav_filename = speech_to_wav(f,os.path.dirname(f)+"/wav",freq)
      output_rows.append(wav_filename)

  if csv_out:
    with open(os.path.dirname(f) + csv_fname, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                              quotechar='|', quoting=csv.QUOTE_MINIMAL)
      for row in output_rows:
        writer.writerow([row])
  else:
    return output_rows


def get_chapters(book):
  import re
  chapters = re.compile("CHAPTER .*\n\n\n").split(book)
  return chapters[1:]

def process_chapters(chapter):
  rows = []
  lines = re.compile('[.?]').split(chapter)
  for line in lines:
    new_line = text_cleaner.english_cleaners(line)
    rows.append([line, new_line])
  return rows

def get_best_match(lines):
  pass

def clean_book(book_file):
  book_text = open(book_file,'r').read()
  chapters = get_chapters(book_text)
  for chapter in chapters:
    lines  = process_chapters(chapter)
    get_best_match(lines)

