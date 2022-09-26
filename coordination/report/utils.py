from typing import List

import json


def generate_chart_script(canvas_id: str, values: List[float]) -> str:
    time_steps = [] if len(values) == 0 else list(range(len(values)))
    datasets = [] if len(values) == 0 else [
        {
            "label": "",
            "data": values,
            "borderColor": "rgba(0,0,255,0.5)",
            "pointRadius": 0
        }
    ]

    data = {
        "labels": time_steps,
        "datasets": datasets
    }

    config = {
        "type": "line",
        "data": data,
        "options": {
            "responsive": True,
            "plugins": {
                "legend": {
                    "display": False,
                },
                "title": {
                    "display": True,
                    "text": ""
                }
            },
            # "scales": {
            #     "y": {
            #         "ticks": {
            #             "callback": "(val) = > (val.toExponential())"
            #         }
            #     }
            # }
        }
    }

    js_string = f"const ctx_{canvas_id} = document.getElementById('{canvas_id}').getContext('2d');\n"
    js_string += f"const chart_{canvas_id} = new Chart(ctx_{canvas_id}, JSON.parse({json.dumps(json.dumps(config))}))"

    return js_string
