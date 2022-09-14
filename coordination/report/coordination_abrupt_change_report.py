import os
from typing import List, Optional, Union, Tuple

import numpy as np
from yattag import Doc, indent

from coordination.audio.audio import AudioSegment, AudioSparseSeries
from coordination.component.speech.vocalics_component import VocalicsSparseSeries

NO_VALUE_STR = "n.a"


class CoordinationAbruptChangeReport:
    def __init__(self, coordination_series: np.ndarray, vocalics_series_a: VocalicsSparseSeries,
                 vocalics_series_b: VocalicsSparseSeries, audio_series_a: Optional[AudioSparseSeries] = None,
                 audio_series_b: Optional[AudioSparseSeries] = None, title: Optional[str] = None):
        self.coordination_series = coordination_series
        self.vocalics_series_a = vocalics_series_a
        self.vocalics_series_b = vocalics_series_b
        self.audio_series_a = audio_series_a
        self.audio_series_b = audio_series_b
        self.title = title

    def export_to_html(self, filepath: str, ignore_under_percentage: float):
        out_dir = os.path.dirname(filepath)
        os.makedirs(f"{out_dir}/audio", exist_ok=True)

        doc, tag, text = Doc().tagtext()

        header_texts, col_spans, data_alignment = CoordinationAbruptChangeReport._get_header()
        table_rows = self._get_rows(ignore_under_percentage)

        with tag("html"):
            doc.stag("link", href="https://fonts.googleapis.com/css?family=Nunito Sans", rel="stylesheet")
            with tag("script"):
                doc.asis(CoordinationAbruptChangeReport._get_js())
            with tag("script", src="https://kit.fontawesome.com/5631021ae3.js", crossorigin="anonymous"):
                pass
            with tag("head"):
                with tag("style"):
                    text(CoordinationAbruptChangeReport._get_style())
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
            ["#", "Timestep", "Previous Coordination", "Current Coordination", "Change", "Series", "Name",
             "Previous Value", "Current Value", "Delay (seconds)", "Previous Utterance", "Current Utterance", "Series",
             "Name", "Previous Value", "Delay (seconds)", "Previous Utterance"]
        ]

        col_spans = [
            [5, 12],
            [5, 7, 5],
            [1] * len(texts[-1]),
        ]

        data_alignment = ["left", "left", "right", "right", "right", "left", "left", "left", "left", "right", "center",
                          "center", "left", "left", "left", "right", "center"]

        return texts, col_spans, data_alignment

    def _get_rows(self, ignore_under_percentage: float) -> List[List[Union[str, AudioSegment]]]:
        def get_vocalic_entries(time_step: int,
                                main_subject_series_name: str,
                                main_subject_vocalic_series: VocalicsSparseSeries,
                                main_subject_previous_value: Optional[np.ndarray],
                                main_subject_previous_time_step: Optional[int],
                                main_subject_audio_series: Optional[AudioSparseSeries],
                                other_subject_series_name: str,
                                other_subject_vocalic_series: VocalicsSparseSeries,
                                other_subject_previous_value: Optional[np.ndarray],
                                other_subject_previous_time_step: Optional[int],
                                other_subject_audio_series: Optional[AudioSparseSeries],
                                ) -> List[Union[str, AudioSegment]]:

            # Main Subject
            main_subject_name = main_subject_vocalic_series.utterances[time_step].subject_id
            if main_subject_previous_value is None:
                main_subject_previous_value = NO_VALUE_STR
                main_subject_delay = NO_VALUE_STR
            else:
                main_subject_previous_value = np.array2string(main_subject_previous_value, precision=2)
                main_subject_delay = str(time_step - main_subject_previous_time_step)

            main_subject_current_value = np.array2string(main_subject_vocalic_series.values[:, time_step], precision=2)

            main_subject_previous_audio = NO_VALUE_STR
            main_subject_current_audio = NO_VALUE_STR
            if main_subject_audio_series is not None and main_subject_previous_time_step is not None and \
                    main_subject_audio_series.audio_segments[main_subject_previous_time_step] is not None:
                main_subject_previous_audio = main_subject_audio_series.audio_segments[main_subject_previous_time_step]

            if main_subject_audio_series is not None:
                main_subject_current_audio = main_subject_audio_series.audio_segments[time_step]

            # Other Subject
            if other_subject_previous_value is None:
                other_subject_name = NO_VALUE_STR
                other_subject_previous_value = NO_VALUE_STR
                other_subject_delay = NO_VALUE_STR
            else:
                other_subject_name = other_subject_vocalic_series.utterances[
                    other_subject_previous_time_step].subject_id
                other_subject_previous_value = np.array2string(other_subject_previous_value, precision=2)
                other_subject_delay = str(time_step - other_subject_previous_time_step)

            other_subject_previous_audio = NO_VALUE_STR
            if other_subject_audio_series is not None and other_subject_previous_time_step is not None and \
                    other_subject_audio_series.audio_segments[other_subject_previous_time_step] is not None:
                other_subject_previous_audio = other_subject_audio_series.audio_segments[
                    other_subject_previous_time_step]

            return [main_subject_series_name, main_subject_name, main_subject_previous_value,
                    main_subject_current_value, main_subject_delay, main_subject_previous_audio,
                    main_subject_current_audio, other_subject_series_name, other_subject_name,
                    other_subject_previous_value, other_subject_delay, other_subject_previous_audio]

        # Get previous timestamps and values of the vocalic series with actual values
        previous_values_same_source_a = self.vocalics_series_a.get_previous_values_same_source()
        previous_values_same_source_b = self.vocalics_series_b.get_previous_values_same_source()
        previous_values_a = self.vocalics_series_a.get_previous_values_same_source()
        previous_values_b = self.vocalics_series_b.get_previous_values_same_source()

        # Only report entries with a significant change in the coordination
        change_rel_magnitude = np.divide(np.diff(self.coordination_series), self.coordination_series[:-1],
                                         out=np.ones_like(self.coordination_series[:-1]) * np.inf,
                                         where=self.coordination_series[:-1] != 0)

        # change_rel_magnitude starts from the second time step, thus we need to add 1 to correct the indexes.
        time_steps = np.where(np.abs(change_rel_magnitude) >= ignore_under_percentage)[0] + 1
        rows: List[List[Union[str, AudioSegment]]] = []
        row_number = 1
        for i, t in enumerate(time_steps):
            time_step = str(t)
            previous_coordination = f"{self.coordination_series[t - 1]:.2f}"
            current_coordination = f"{self.coordination_series[t]:.2f}"
            coordination_rel_change = f"{change_rel_magnitude[t - 1] * 100:.2f}%"

            row_common = [str(row_number), time_step, previous_coordination, current_coordination, coordination_rel_change]

            if self.vocalics_series_a.mask[t] == 1 or self.vocalics_series_b.mask[t] == 1:
                row_vocalics_a = None
                row_vocalics_b = None

                if self.vocalics_series_a.mask[t] == 1:
                    main_previous_time, main_previous_value = (None, None)
                    if previous_values_same_source_a[t] is not None:
                        main_previous_time, main_previous_value = previous_values_same_source_a[t]
                    other_previous_time, other_previous_value = (None, None)
                    if previous_values_b[t] is not None:
                        other_previous_time, other_previous_value = previous_values_b[t]

                    if main_previous_value is not None and other_previous_value is not None:
                        # If there's any change when main or other previous value is None, this is due to coordination
                        # drifting, and we do add that to the report.
                        row_vocalics_a = get_vocalic_entries(t, "A", self.vocalics_series_a, main_previous_value,
                                                             main_previous_time, self.audio_series_a, "B",
                                                             self.vocalics_series_b, other_previous_value,
                                                             other_previous_time, self.audio_series_b)

                if self.vocalics_series_b.mask[t] == 1:
                    main_previous_time, main_previous_value = (None, None)
                    if previous_values_same_source_b[t] is not None:
                        main_previous_time, main_previous_value = previous_values_same_source_b[t]
                    other_previous_time, other_previous_value = (None, None)
                    if previous_values_a[t] is not None:
                        other_previous_time, other_previous_value = previous_values_a[t]

                    if main_previous_value is not None and other_previous_value is not None:
                        # If there's any change when main or other previous value is None, this is due to coordination
                        # drifting, and we do add that to the report.
                        row_vocalics_b = get_vocalic_entries(t, "B", self.vocalics_series_b, main_previous_value,
                                                             main_previous_time, self.audio_series_b, "A",
                                                             self.vocalics_series_a, other_previous_value,
                                                             other_previous_time, self.audio_series_a)

                if row_vocalics_a is not None:
                    rows.append(row_common + row_vocalics_a)
                    if row_vocalics_b is not None:
                        # Do not repeat common information
                        rows.append([""] * len(row_common) + row_vocalics_b)
                    row_number += 1
                elif row_vocalics_b is not None:
                    rows.append(row_common + row_vocalics_b)
                    row_number += 1

            else:
                # Change is due to drifting only. We don't add that to the report to avoid clutter.
                # rows.append(row_common + [NO_VALUE_STR] * 12)
                pass

        return rows
