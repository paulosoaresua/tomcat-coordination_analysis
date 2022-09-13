import os
from typing import List, Optional, Union, Tuple

from datetime import datetime

from yattag import Doc, indent

from coordination.audio.audio import AudioSegment, TrialAudio
from coordination.entity.trial import Trial
from coordination.entity.vocalics import Utterance

NO_VALUE_STR = "n.a"


class AudioAlignmentReport:
    def __init__(self, trial: Trial, audio: TrialAudio, title: Optional[str] = None):
        self.trial = trial
        self.audio = audio
        self.title = title

    def export_to_html(self, filepath: str):
        out_dir = os.path.dirname(filepath)
        os.makedirs(f"{out_dir}/audio", exist_ok=True)

        doc, tag, text = Doc().tagtext()

        header_texts, col_spans, data_alignment = AudioAlignmentReport._get_header()
        table_rows = self._get_rows()

        with tag("html"):
            doc.stag("link", href="https://fonts.googleapis.com/css?family=Nunito Sans", rel="stylesheet")
            with tag("script"):
                doc.asis(AudioAlignmentReport._get_js())
            with tag("script", src="https://kit.fontawesome.com/5631021ae3.js", crossorigin="anonymous"):
                pass
            with tag("head"):
                with tag("style"):
                    text(AudioAlignmentReport._get_style())
            with tag("body"):
                if self.title is not None:
                    with tag("h1"):
                        text(self.title)
                with tag("table", klass="styled-table"):
                    with tag("thead"):
                        for i, header_row in enumerate(header_texts):
                            with tag("tr"):
                                for j, header_cell in enumerate(header_row):
                                    if col_spans[i][j] > 1:
                                        with tag("th", colspan=col_spans[i][j]):
                                            text(header_cell)
                                    else:
                                        with tag("th"):
                                            text(header_cell)

                    with tag("tbody"):
                        for i, table_row in enumerate(table_rows):
                            with tag("tr"):
                                for j, table_cell in enumerate(table_row):
                                    with tag("td", style=f"text-align:{data_alignment[j]}"):
                                        if isinstance(table_cell, AudioSegment):
                                            audio_src = f"{out_dir}/audio/audio_{i}_{j}.wav"
                                            # table_cell.save_to_mp3(audio_src, False)
                                            table_cell.save_to_wav(audio_src)
                                            with tag("div", klass="tooltip"):
                                                with tag("a", href="#!", klass="play-button",
                                                         onclick=f"playAudio('./audio/audio_{i}_{j}.wav')"):
                                                    with tag("i", klass="fa fa-play fa-2x"):
                                                        pass
                                                # with tag("span", klass="tooltiptext"):
                                                #     text(
                                                #         f"{table_cell.transcription}: [{table_cell.start.isoformat()}, {table_cell.end.isoformat()}]")
                                        else:
                                            text(table_cell)

        with open(filepath, "w") as f:
            f.write(indent(doc.getvalue()))

    @staticmethod
    def _get_js() -> str:
        js = """
            function playAudio(source) { 
                console.log(source)
                var audio = new Audio(source); 
                audio.play()
                    .then(() => {
                        // Audio is playing.
                    })
                    .catch(error => { 
                        console.log(error); 
                    }); 
            }
        """

        return js

    @staticmethod
    def _get_style() -> str:
        css = """
              body {
                  font-family: 'Nunito Sans';font-size: 14px;
              }
        
              .styled-table {
                  border-collapse: collapse;
                  margin: 25px 0;
                  font-size: 0.9em;
                  font-family: sans-serif;
                  min-width: 400px;
                  box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);                  
              }
              
              .styled-table th, td {
                  border-left: 1px solid #dddddd;
                  border-right: 1px solid #dddddd;
              }
              
              .styled-table thead tr {
                  background-color: #009879;
                  color: #ffffff;
                  text-align: center;     
                  border-bottom: 1px solid #dddddd;          
              }

              .styled-table th,
              .styled-table td {
                  padding: 10px 12px;
              }

              .styled-table tbody tr {
                  border-bottom: 1px solid #dddddd;
              }

              .styled-table tbody tr:nth-of-type(even) {
                  background-color: #f3f3f3;
              }

              .styled-table tbody tr:last-of-type {
                  border-bottom: 2px solid #009879;
              }

              .styled-table tbody tr:hover {
                  background-color: #3BF7D2;
              }
              
              .play-button {
                  box-sizing: border-box;
                  display:inline-block;
                  width:20px;
                  height:20px;
                  padding-top: 3px;
                  padding-left: 2px;
                  line-height: 18px;
                  border: 2px solid #fff;
                  border-radius: 50%;
                  color:#f5f5f5;
                  text-align:center;
                  text-decoration:none;
                  background-color: rgba(0,0,0,0.6);
                  font-size:5px;
                  font-weight:bold;
                  transition: all 0.3s ease;
            }
            
            .play-button:hover {
                  background-color: rgba(0,0,0,0.8);
                  box-shadow: 0px 0px 10px rgba(255,255,100,1);
                  text-shadow: 0px 0px 10px rgba(255,255,100,1);
            }
            
            .tooltip .tooltiptext {
              visibility: hidden;
              width: 120px;
              background-color: black;
              color: #fff;
              text-align: center;
              border-radius: 6px;
              padding: 5px 0;
            
              /* Position the tooltip */
              position: absolute;
              z-index: 1;
            }

            .tooltip:hover .tooltiptext {
              visibility: visible;
            }
        """

        return css

    @staticmethod
    def _get_header() -> Tuple[List[List[str]], List[List[int]], List[str]]:
        texts = [
            ["#", "Duration Since Trial Start", "Duration Since Mission Start", "Player", "Start", "End",
             "Duration (seconds)", "Utterance", "Audio"]
        ]

        col_spans = [
            [1] * len(texts[-1]),
        ]

        data_alignment = ["left", "center", "center", "center", "center", "center", "center", "left", "center"]

        return texts, col_spans, data_alignment

    def _get_rows(self) -> List[List[Union[str, AudioSegment]]]:
        # List of all utterances sorted by start time
        utterances: List[Utterance] = []
        for u in self.trial.vocalics.utterances_per_subject.values():
            utterances.extend(u)
        utterances.sort(key=lambda utterance: utterance.start)

        entries = []
        for i, utterance in enumerate(utterances):
            voice = self.audio.audio_per_participant[utterance.subject_id]
            entry = [
                str(i + 1),
                AudioAlignmentReport._get_relative_time(utterance.start, self.trial.metadata.trial_start),
                AudioAlignmentReport._get_relative_time(utterance.start, self.trial.metadata.mission_start),
                utterance.subject_id,
                utterance.start.isoformat(),
                utterance.end.isoformat(),
                str((utterance.end - utterance.start).total_seconds()),
                utterance.text,
                AudioSegment(utterance.subject_id, utterance.start, utterance.end,
                             voice.get_data_segment(utterance.start, utterance.end), voice.sample_rate)
            ]

            entries.append(entry)

        return entries

    @staticmethod
    def _get_relative_time(time: datetime, start_time: datetime):
        seconds = (time - start_time).total_seconds()
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)

        minutes = str(minutes) if minutes > 9 else f"0{minutes}"
        seconds = str(seconds) if seconds > 9 else f"0{seconds}"

        return f"{minutes}:{seconds}"
