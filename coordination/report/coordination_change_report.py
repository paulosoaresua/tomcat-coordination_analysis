import os
from typing import List, Optional, Union, Tuple

import numpy as np
from yattag import Doc, indent

from coordination.audio.audio import AudioSegment, TrialAudio
from coordination.component.speech.vocalics_component import VocalicsSparseSeries

NO_VALUE_STR = "n.a"


class CoordinationChangeReport:
    def __init__(self, coordination_series: np.ndarray, vocalic_series: VocalicsSparseSeries,
                 trial_audio: Optional[TrialAudio] = None, title: Optional[str] = None):
        self.coordination_series = coordination_series
        self.vocalic_series = vocalic_series
        self.trial_audio = trial_audio
        self.title = title

    def export_to_html(self, filepath: str, ignore_under_percentage: float):
        out_dir = os.path.dirname(filepath)
        os.makedirs(f"{out_dir}/audio", exist_ok=True)

        doc, tag, text = Doc().tagtext()

        header_texts, col_spans, data_alignment = CoordinationChangeReport._get_header()
        table_rows = self._get_rows(ignore_under_percentage)

        with tag("html"):
            doc.stag("link", href="https://fonts.googleapis.com/css?family=Nunito Sans", rel="stylesheet")
            with tag("script"):
                doc.asis(CoordinationChangeReport._get_js())
            with tag("script", src="https://kit.fontawesome.com/5631021ae3.js", crossorigin="anonymous"):
                pass
            with tag("head"):
                with tag("style"):
                    text(CoordinationChangeReport._get_style())
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
                                                with tag("span", klass="play_button_tooltip"):
                                                    text(table_cell.transcription)
                                                    doc.stag("br")
                                                    doc.stag("br")
                                                    text(table_cell.start.isoformat())
                                                    doc.stag("br")
                                                    text(table_cell.end.isoformat())
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
            
            .tooltip .play_button_tooltip {
              visibility: hidden;
              width: 250px;
              background-color: black;
              color: #fff;
              text-align: center;
              border-radius: 6px;
              padding: 5px 0;
            
              /* Position the tooltip */
              position: absolute;
              z-index: 1;
            }

            .tooltip:hover .play_button_tooltip {
              visibility: visible;
            }
        """

        return css

    @staticmethod
    def _get_header() -> Tuple[List[List[str]], List[List[int]], List[str]]:
        texts = [
            ["", "Vocalics"],
            ["", "Main Subject", "Other Subject"],
            ["#", "Timestep", "Previous Coordination", "Current Coordination", "Change", "Name",
             "Previous Value", "Current Value", "Delay (seconds)", "Previous Utterance", "Current Utterance",
             "Name", "Previous Value", "Delay (seconds)", "Previous Utterance"]
        ]

        col_spans = [
            [5, 10],
            [5, 6, 5],
            [1] * len(texts[-1]),
        ]

        data_alignment = ["left", "left", "right", "right", "right", "left", "left", "left", "right", "center",
                          "center", "left", "left", "right", "center"]

        return texts, col_spans, data_alignment

    def _get_rows(self, ignore_under_percentage: float) -> List[List[Union[str, AudioSegment]]]:
        # Only report entries with a significant change in the coordination
        change_rel_magnitude = np.divide(np.diff(self.coordination_series), self.coordination_series[:-1],
                                         out=np.ones_like(self.coordination_series[:-1]) * np.inf,
                                         where=self.coordination_series[:-1] != 0)

        # change_rel_magnitude starts from the second time step, thus we need to add 1 to correct the indexes.
        time_steps = np.where(np.abs(change_rel_magnitude) >= ignore_under_percentage)[0] + 1
        rows: List[List[Union[str, AudioSegment]]] = []
        for i, t in enumerate(time_steps):
            row_number = str(len(rows) + 1)
            time_step = str(t)
            previous_coordination = f"{self.coordination_series[t - 1]:.2f}"
            current_coordination = f"{self.coordination_series[t]:.2f}"
            coordination_rel_change = f"{change_rel_magnitude[t - 1] * 100:.2f}%"

            row = [str(row_number), time_step, previous_coordination, current_coordination,
                   coordination_rel_change]

            if self.vocalic_series.mask[t] == 1:
                main_previous_time, main_previous_value = (None, None)
                if self.vocalic_series.previous_from_self[t] is not None:
                    main_previous_time = self.vocalic_series.previous_from_self[t]
                    main_previous_value = self.vocalic_series.values[:, main_previous_time]
                other_previous_time, other_previous_value = (None, None)
                if self.vocalic_series.previous_from_other[t] is not None:
                    other_previous_time = self.vocalic_series.previous_from_other[t]
                    other_previous_value = self.vocalic_series.values[:, other_previous_time]

                if main_previous_time is not None and other_previous_time is not None:
                    # If there's any change when main or other previous value is None, this is due to coordination
                    # drifting, and we do add that to the report.

                    # Main Subject
                    main_subject_current_value = np.array2string(self.vocalic_series.values[:, t], precision=2)
                    main_subject_name = self.vocalic_series.utterances[t].subject_id
                    main_subject_previous_value = np.array2string(main_previous_value, precision=2)
                    main_subject_delay = str(t - main_previous_time)

                    main_subject_previous_audio = NO_VALUE_STR
                    main_subject_current_audio = NO_VALUE_STR
                    if self.trial_audio is not None:
                        main_subject_current_audio = self.trial_audio.audio_per_participant[
                            main_subject_name].get_audio_segment_from_utterance(self.vocalic_series.utterances[t])
                        if main_previous_time is not None:
                            main_subject_previous_audio = self.trial_audio.audio_per_participant[
                                main_subject_name].get_audio_segment_from_utterance(
                                self.vocalic_series.utterances[main_previous_time])

                    # Other Subject
                    other_subject_name = self.vocalic_series.utterances[other_previous_time].subject_id
                    other_subject_previous_value = np.array2string(other_previous_value, precision=2)
                    other_subject_delay = str(t - other_previous_time)

                    other_subject_previous_audio = NO_VALUE_STR
                    if self.trial_audio is not None and other_previous_time is not None:
                        other_subject_previous_audio = self.trial_audio.audio_per_participant[
                            other_subject_name].get_audio_segment_from_utterance(
                            self.vocalic_series.utterances[other_previous_time])

                    row_vocalics = [main_subject_name, main_subject_previous_value, main_subject_current_value,
                                    main_subject_delay, main_subject_previous_audio, main_subject_current_audio,
                                    other_subject_name, other_subject_previous_value, other_subject_delay,
                                    other_subject_previous_audio]

                    row.extend(row_vocalics)
                    rows.append(row)

            else:
                # Change is due to drifting only. We don't add that to the report to avoid clutter.
                # rows.append(row_common + [NO_VALUE_STR] * 12)
                pass

        return rows
