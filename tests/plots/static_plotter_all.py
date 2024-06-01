import os

import torch
from plotly import graph_objects as go
from plots._utils import (
    angle_to_coord,
    compute_2d_non_conflicting_cone,
    make_cone_scatter,
    make_polygon_scatter,
    make_right_angle,
    make_segment_scatter,
    make_vector_scatter,
)

from torchjd.aggregation import DualProjWrapper, MeanWeighting, MGDAWeighting, WeightedAggregator

PLOT_CONSTRUCTION_SEGMENTS = True
RIGHT_ANGLE_SIZE = 0.07


def main():
    angle1 = 2.6
    angle2 = 0.3277
    norm1 = 0.9
    norm2 = 2.8
    g1 = torch.tensor(angle_to_coord(angle1, norm1))
    g2 = torch.tensor(angle_to_coord(angle2, norm2))
    matrix = torch.stack([g1, g2])

    aggregators = {
        "Mean": WeightedAggregator(MeanWeighting()),
        "DualProj": WeightedAggregator(DualProjWrapper(MeanWeighting())),
        "MGDA": WeightedAggregator(MGDAWeighting()),
    }
    name_to_label_location = {
        "Mean": "top center",
        "DualProj": "top center",
        "MGDA": "top center",
    }
    results = {name: aggregator(matrix) for name, aggregator in aggregators.items()}

    fig = go.Figure()

    start_angle, opening = compute_2d_non_conflicting_cone(matrix.numpy())
    cone = make_cone_scatter(start_angle, opening, label="Non-conflicting cone", printable=False)
    fig.add_trace(cone)

    if PLOT_CONSTRUCTION_SEGMENTS:
        g1_g2_segment = make_segment_scatter(g1, g2)
        dual_proj_segment = make_segment_scatter(results["Mean"], results["DualProj"])

        mgda_right_angle = make_polygon_scatter(
            make_right_angle(
                results["MGDA"], size=RIGHT_ANGLE_SIZE, positive_para=False, positive_orth=False
            )
        )
        dual_proj_right_angle = make_polygon_scatter(
            make_right_angle(
                results["DualProj"], size=RIGHT_ANGLE_SIZE, positive_orth=False, positive_para=False
            )
        )

        fig.add_trace(g1_g2_segment)
        fig.add_trace(dual_proj_segment)

        fig.add_trace(mgda_right_angle)
        fig.add_trace(dual_proj_right_angle)

    for i in range(len(matrix)):
        label = r"$\Huge{" + f"g_{i + 1}" + r"}$"

        gradient_scatter = make_vector_scatter(
            matrix[i],
            color="rgb(40, 40, 40)",
            label=label,
            showlegend=False,
            dash=False,
            textposition="bottom center",
            text_size=32,
            marker_size=22,
            line_width=4,
        )
        fig.add_trace(gradient_scatter)

    for name, result in results.items():
        update_scatter = make_vector_scatter(
            result,
            color="rgb(0, 0, 215)",
            label=name,
            textposition=name_to_label_location[name],
            showlegend=False,
            dash=False,
            text_size=32,
            marker_size=22,
            line_width=4,
        )
        fig.add_trace(update_scatter)

    fig.update_layout(
        hovermode=False,
        width=912,
        height=528,
        plot_bgcolor="white",
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        range=[-0.95, 2.85],
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    fig.update_yaxes(range=[-0.1, 2.1], showgrid=False, zeroline=False, visible=False)
    try:
        os.makedirs("images/")
    except FileExistsError:
        pass

    fig.write_image("images/directions.svg")

    fig.show()


if __name__ == "__main__":
    main()
