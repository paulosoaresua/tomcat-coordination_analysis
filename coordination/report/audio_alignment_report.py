import os
from typing import List, Optional, Union, Tuple

from datetime import datetime
from pkg_resources import resource_string

import numpy as np
from yattag import Doc, indent

from coordination.audio.audio import AudioSegment, TrialAudio
from coordination.entity.trial import Trial
from coordination.entity.vocalics import Utterance
from coordination.report.utils import generate_chart_script

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
            with tag("script", src="https://cdn.jsdelivr.net/npm/chart.js"):
                pass
            with tag("script", src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"):
                pass
            with tag("head"):
                with tag("style"):
                    text(AudioAlignmentReport._get_style())
            with tag("body"):
                if self.title is not None:
                    with tag("h1"):
                        text(self.title)
                self._build_subtitle(doc, tag, text)
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
                                        elif isinstance(table_cell, list):
                                            canvas_id = f"vocalics_plot_{i}_{j}"
                                            doc.stag("canvas", id=canvas_id, width="400", height="200")
                                            with tag("script"):
                                                doc.asis(generate_chart_script(canvas_id, table_cell))
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
        css = resource_string("coordination.resources.style", "report.css").decode()
        css += resource_string("coordination.resources.style", "play_button.css").decode()
        tooltip_css = """                   
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

        css += tooltip_css

        return css

    @staticmethod
    def _get_header() -> Tuple[List[List[str]], List[List[int]], List[str]]:
        texts = [
            ["#", "Duration Since Trial Start", "Duration Since Mission Start", "Player", "Start", "End",
             "Duration (seconds)", "Utterance", "Audio", "Pitch", "Intensity"]
        ]

        col_spans = [
            [1] * len(texts[-1]),
        ]

        data_alignment = ["left", "center", "center", "center", "center", "center", "center", "left", "center",
                          "center", "center"]

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
            pitches = [] if utterance.vocalic_series.size == 0 else utterance.vocalic_series.values[0].tolist()
            intensities = [] if utterance.vocalic_series.size == 0 else utterance.vocalic_series.values[1].tolist()
            entry = [
                str(i + 1),
                AudioAlignmentReport._get_relative_time(utterance.start, self.trial.metadata.trial_start),
                AudioAlignmentReport._get_relative_time(utterance.start, self.trial.metadata.mission_start),
                utterance.subject_id,
                utterance.start.isoformat(),
                utterance.end.isoformat(),
                f"{((utterance.end - utterance.start).total_seconds()):.2f}",
                utterance.text,
                AudioSegment(utterance.subject_id, utterance.start, utterance.end,
                             voice.get_data_segment(utterance.start, utterance.end), voice.sample_rate),
                pitches,
                intensities
            ]

            entries.append(entry)

        return entries

    @staticmethod
    def _get_relative_time(time: datetime, start_time: datetime):
        seconds = int((time - start_time).total_seconds())
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)

        minutes = str(minutes) if minutes > 9 else f"0{minutes}"
        seconds = str(seconds) if seconds > 9 else f"0{seconds}"

        return f"{minutes}:{seconds}"

    def _build_subtitle(self, doc, tag, text):
        # List of all utterances sorted by start time
        utterances: List[Utterance] = []
        for u in self.trial.vocalics.utterances_per_subject.values():
            utterances.extend(u)
        utterances.sort(key=lambda utterance: utterance.start)

        null_vocalics = 0
        subjects = set()
        for i, utterance in enumerate(utterances):
            if utterance.vocalic_series.size == 0 or np.all(utterance.vocalic_series.values.var(axis=1) < 1E-16):
                null_vocalics += 1
                subjects.add(utterance.subject_id)

        with tag("ul"):
            with tag("li"):
                text(f"{len(utterances)} utterances")

            with tag("li"):
                subjects = ", ".join(sorted(subjects))
                text(f"{null_vocalics} null vocalics - [ {subjects} ]")
