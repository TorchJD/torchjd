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
    project,
)

from torchjd.aggregation import MeanWeighting, UPGradWrapper, WeightedAggregator

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

    aggregators = {"UPGrad": WeightedAggregator(UPGradWrapper(MeanWeighting()))}
    results = {name: aggregator(matrix) for name, aggregator in aggregators.items()}

    fig = go.Figure()

    start_angle, opening = compute_2d_non_conflicting_cone(matrix.numpy())
    cone = make_cone_scatter(start_angle, opening, label="Non-conflicting cone", printable=False)
    fig.add_trace(cone)

    if PLOT_CONSTRUCTION_SEGMENTS:
        g1_proj = g1 - project(g1, onto=g2)
        g2_proj = g2 - project(g2, onto=g1)

        g1_proj_segment = make_segment_scatter(g1, g1_proj)
        g2_proj_segment = make_segment_scatter(g2, g2_proj)
        origin_g1_proj_vector = make_vector_scatter(
            g1_proj,
            color="rgb(100, 100, 100)",
            label=r"$\Huge{" + r"\pi_J(g_1)" + r"}$",
            line_width=3,
            marker_size=16,
            textposition="top left",
        )
        origin_g2_proj_vector = make_vector_scatter(
            g2_proj,
            color="rgb(100, 100, 100)",
            label=r"$\Huge{" + r"\pi_J(g_2)" + r"}$",
            line_width=3,
            marker_size=16,
            textposition="top right",
        )

        g1_proj_upgrad_segment = make_segment_scatter(g1_proj, results["UPGrad"])
        g2_proj_upgrad_segment = make_segment_scatter(g2_proj, results["UPGrad"])

        g1_proj_right_angle = make_polygon_scatter(
            make_right_angle(g1_proj, size=RIGHT_ANGLE_SIZE, positive_para=False)
        )
        g2_proj_right_angle = make_polygon_scatter(
            make_right_angle(
                g2_proj, size=RIGHT_ANGLE_SIZE, positive_orth=False, positive_para=False
            )
        )

        fig.add_trace(g1_proj_segment)
        fig.add_trace(g2_proj_segment)

        fig.add_trace(g1_proj_upgrad_segment)
        fig.add_trace(g2_proj_upgrad_segment)

        fig.add_trace(g1_proj_right_angle)
        fig.add_trace(g2_proj_right_angle)

        fig.add_trace(origin_g1_proj_vector)
        fig.add_trace(origin_g2_proj_vector)

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
            textposition="top center",
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

    fig.write_image("images/direction_upgrad.svg")

    fig.show()


if __name__ == "__main__":
    main()
