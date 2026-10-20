"""
Data writer for geometry problem generation pipeline.

Handles asynchronous figure drawing and data writing with Ray.
"""

import json
import logging
import os

import ray
from PIL import Image, ImageOps

POINT_COORDS_GRID_SIZE = 256


def convert_svg_to_png(svg_path, png_path, width=1024):
    """Convert SVG file to PNG format."""
    # Lazy import to avoid cairo dependency when not needed
    import cairosvg

    output_dir = os.path.dirname(png_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=width)
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert '{svg_path}' to PNG. Error: {str(e)}"
        ) from e


def save_figure_as_png(
    fig,
    png_path: str,
    img_pixels: int,
    direct_png: bool,
    svg_path: str | None = None,
):
    """Persist a matplotlib figure to PNG using the selected rendering path."""
    output_dir = os.path.dirname(png_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if direct_png:
        width_inches = fig.get_size_inches()[0]
        dpi = img_pixels / width_inches
        fig.savefig(
            png_path,
            format="png",
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        return

    if svg_path is None:
        raise ValueError("svg_path is required when direct_png is False")

    svg_output_dir = os.path.dirname(svg_path)
    if svg_output_dir and not os.path.exists(svg_output_dir):
        os.makedirs(svg_output_dir, exist_ok=True)

    fig.savefig(
        svg_path,
        format="svg",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    convert_svg_to_png(svg_path, png_path, width=img_pixels)


def invert_image_file(image_path: str) -> None:
    """Invert an image file in place while preserving alpha channels."""
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            r, g, b, a = img.split()
            inverted_rgb = ImageOps.invert(Image.merge("RGB", (r, g, b)))
            r_inv, g_inv, b_inv = inverted_rgb.split()
            img_out = Image.merge("RGBA", (r_inv, g_inv, b_inv, a))
        elif img.mode == "LA":
            lightness, alpha = img.split()
            img_out = Image.merge("LA", (ImageOps.invert(lightness), alpha))
        else:
            img_out = ImageOps.invert(img.convert("RGB"))
        img_out.save(image_path)


def _clamp_grid_coord(value: int, grid_size: int) -> int:
    return max(0, min(value, grid_size - 1))


def build_point_coords_grid(
    display_points: dict[str, tuple[float, float]],
    mapping: dict[str, str] | None,
    canvas_width: float,
    canvas_height: float,
    grid_size: int = POINT_COORDS_GRID_SIZE,
) -> dict[str, list[int]]:
    """Convert rendered display coordinates into a fixed top-left grid."""
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("canvas dimensions must be positive")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    output: dict[str, list[int]] = {}
    if mapping is None:
        sorted_items = [(name, name) for name in sorted(display_points)]
    else:
        sorted_items = sorted(mapping.items())
    for source_name, target_name in sorted_items:
        if source_name not in display_points:
            continue
        display_x, display_y = display_points[source_name]
        gx = _clamp_grid_coord(
            round(display_x / canvas_width * (grid_size - 1)),
            grid_size,
        )
        gy = _clamp_grid_coord(
            round((1.0 - display_y / canvas_height) * (grid_size - 1)),
            grid_size,
        )
        output[target_name] = [gx, gy]
    return output


def extract_point_coords_grid(
    ax,
    point_lookup: dict[str, object],
    mapping: dict[str, str] | None,
    grid_size: int = POINT_COORDS_GRID_SIZE,
) -> dict[str, list[int]]:
    """Project rendered points into the fixed image-aligned grid."""
    ax.figure.canvas.draw()
    canvas_width, canvas_height = ax.figure.canvas.get_width_height()
    display_points: dict[str, tuple[float, float]] = {}
    for name, point in point_lookup.items():
        point_num = getattr(point, "num", None)
        if point_num is None:
            continue
        display_x, display_y = ax.transData.transform((point_num.x, point_num.y))
        display_points[name] = (float(display_x), float(display_y))

    return build_point_coords_grid(
        display_points,
        mapping,
        canvas_width,
        canvas_height,
        grid_size=grid_size,
    )


@ray.remote(num_cpus=0.5)
def draw_figure_task(
    draw_data: dict,
    defs_data: dict,
    imgs_dir: str,
    imgs_png_dir: str,
    file_idx: int,
    session_id: str,
    img_mode: int,
    direct_png: bool,
    img_pixels: int,
):
    """
    Ray remote task for drawing figures.

    Args:
        draw_data: Drawing metadata (clauses, mapping, goal, coords, seed)
        defs_data: Serialized definitions
        imgs_dir: SVG output directory
        imgs_png_dir: PNG output directory
        file_idx: File index for naming
        session_id: Unique session identifier
        img_mode: 0=no images, 1=with annotations only, 2=without annotations only, 3=both
        direct_png: Whether to save PNG directly instead of svg -> png conversion
        img_pixels: Output image width in pixels

    Returns:
        Tuple of (file_idx, paths_update, error)
    """
    try:
        from newclid.algebraic_reasoning.algebraic_manipulator import (
            AlgebraicManipulator,
        )
        from newclid.dependencies.dependency_graph import DependencyGraph
        from newclid.dependencies.symbols import Point
        from newclid.numerical.geometries import PointNum
        from newclid.numerical.draw_figure import init_figure
        from newclid.numerical.draw_clause_figure import draw_clauses
        from newclid.statement import Statement
        from newclid.formulations.clause import Clause
        from newclid.formulations.definition import DefinitionJGEX
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Rebuild defs from serialized data
        defs = {k: DefinitionJGEX(**v) for k, v in defs_data.items()}

        # Rebuild Point objects with coordinates
        dep_graph = DependencyGraph(AlgebraicManipulator())
        for name, (x, y) in draw_data["point_coords"].items():
            point = Point(name=name, symbols_graph=dep_graph.symbols_graph, dep=None)
            point.num = PointNum(x, y)
            dep_graph.symbols_graph.name2node[name] = point

        # Rebuild Clause objects from (points, sentences) tuples
        clauses = [
            Clause(points=tuple(c[0]), sentences=tuple(tuple(s) for s in c[1]))
            for c in draw_data["clauses"]
        ]

        # Rebuild goal Statement
        goal = Statement.from_tokens(tuple(draw_data["goal_tokens"]), dep_graph)

        paths_update = {}

        # Determine which images to generate based on img_mode
        draw_configs = []
        if img_mode == 1:
            draw_configs = [("", True)]
        elif img_mode == 2:
            draw_configs = [("_no_annotations", False)]
        elif img_mode == 3:
            draw_configs = [("", True), ("_no_annotations", False)]

        for suffix, annotations in draw_configs:
            file_name = f"{session_id}_{file_idx}{suffix}"
            svg_path = os.path.join(imgs_dir, f"{file_name}.svg")
            png_path = os.path.join(imgs_png_dir, f"{file_name}.png")

            # Draw figure
            fig = init_figure()
            draw_clauses(
                fig.axes[0],
                clauses,
                defs,
                goal,
                np.random.default_rng(draw_data["seed"]),
                dep_graph,
                draw_data["mapping"],
                annotations,
            )
            if "point_coords_grid" not in paths_update:
                premise_mapping = {
                    point_name: draw_data["mapping"][point_name]
                    for point_name in draw_data["premise_point_names"]
                    if point_name in draw_data["mapping"]
                }
                paths_update["point_coords_grid"] = extract_point_coords_grid(
                    fig.axes[0],
                    dep_graph.symbols_graph.name2node,
                    premise_mapping,
                )
            save_figure_as_png(
                fig,
                png_path=png_path,
                img_pixels=img_pixels,
                direct_png=direct_png,
                svg_path=svg_path,
            )
            plt.close(fig)

            paths_update[f"image_path{suffix}"] = png_path

        return file_idx, paths_update, None
    except Exception:
        import traceback

        return file_idx, {}, traceback.format_exc()


class Writer:
    """
    Handles data writing and asynchronous figure drawing.

    Manages pending draw tasks and writes completed data to JSONL files.
    """

    def __init__(
        self,
        output_dir: str,
        file_prefix: str,
        img_mode: int,
        defs_data: dict,
        session_id: str,
        direct_png: bool = True,
        img_pixels: int = 512,
    ):
        """
        Initialize writer.

        Args:
            output_dir: Output directory (with date)
            file_prefix: Filename prefix for output files
            img_mode: Image generation mode (0-3)
            defs_data: Serialized definitions for drawing
            session_id: Unique session identifier
            direct_png: Whether to save PNG directly instead of svg -> png conversion
            img_pixels: Output image width in pixels
        """
        if img_pixels <= 0:
            raise ValueError("img_pixels must be a positive integer")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_prefix = file_prefix
        self.img_mode = img_mode
        self.defs_data = defs_data
        self.session_id = session_id
        self.direct_png = direct_png
        self.img_pixels = img_pixels

        self.data_count = 0
        self.written_count = 0
        self.pending_draw_tasks = {}  # task_id -> (file_idx, data_item)
        self.pending_write_data = {}  # file_idx -> result_data (completed draw, pending write)

    def clear(self):
        """Remove existing output files."""
        import shutil

        filename = os.path.join(self.output_dir, self.file_prefix + ".jsonl")
        imgs_dir = os.path.join(self.output_dir, "imgs")
        imgs_png_dir = os.path.join(self.output_dir, "imgs_png")
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(imgs_dir):
            shutil.rmtree(imgs_dir)
        if os.path.exists(imgs_png_dir):
            shutil.rmtree(imgs_png_dir)

    def _process_completed_draw_tasks(self, wait_all: bool = False):
        """Process completed draw tasks and store results."""
        if not self.pending_draw_tasks:
            return

        # Get tasks to process
        if wait_all:
            done_tasks = list(self.pending_draw_tasks.keys())
        else:
            done_tasks, _ = ray.wait(
                list(self.pending_draw_tasks.keys()),
                num_returns=len(self.pending_draw_tasks),
                timeout=0,
            )

        # Process each completed task
        for task_id in done_tasks:
            try:
                file_idx, paths_update, error = ray.get(task_id)
                if error:
                    logging.warning(f"Draw task {file_idx} failed: {error}")
                else:
                    data_item = self.pending_draw_tasks[task_id][1]
                    self.pending_write_data[file_idx] = {**paths_update, **data_item}
            except Exception as e:
                logging.error(f"Failed to get draw task result: {e}")
            del self.pending_draw_tasks[task_id]

    def _flush_completed_data(self, filename: str) -> int:
        """Write completed data to file in order. Returns number of items written."""
        if not self.pending_write_data:
            return 0

        # Sort by file_idx and write in order
        sorted_indices = sorted(self.pending_write_data.keys())
        with open(filename, "a", encoding="utf-8") as f:
            for idx in sorted_indices:
                result_data = self.pending_write_data[idx]
                json.dump(result_data, f, ensure_ascii=False)
                f.write("\n")
        written = len(self.pending_write_data)
        self.written_count += written
        self.pending_write_data.clear()
        return written

    def write_data(self, all_data: list, force: bool = False):
        """
        Append data to buffer and submit draw tasks if needed.

        Args:
            all_data: List of data items to write
            force: If True, wait for all pending tasks and flush immediately
        """
        filename = os.path.join(self.output_dir, self.file_prefix + ".jsonl")
        imgs_dir = os.path.join(self.output_dir, "imgs")
        imgs_png_dir = os.path.join(self.output_dir, "imgs_png")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os.makedirs(imgs_png_dir, exist_ok=True)
        if not self.direct_png:
            os.makedirs(imgs_dir, exist_ok=True)

        for data_item in all_data:
            self.data_count += 1
            data_item.pop("_timings", None)
            if self.img_mode > 0 and "draw_data" in data_item:
                draw_data = data_item.pop("draw_data")
                # Submit draw task to Ray
                task_id = draw_figure_task.remote(
                    draw_data,
                    self.defs_data,
                    imgs_dir,
                    imgs_png_dir,
                    self.data_count,
                    self.session_id,
                    self.img_mode,
                    self.direct_png,
                    self.img_pixels,
                )
                self.pending_draw_tasks[task_id] = (self.data_count, data_item)
            else:
                # No image needed, write directly
                data_item.pop("draw_data", None)
                self.pending_write_data[self.data_count] = data_item

        # Process any completed draw tasks
        self._process_completed_draw_tasks(wait_all=False)

        # Flush completed data periodically or when forced
        should_flush = (
            len(self.pending_write_data) > 1000
            or len(self.pending_draw_tasks) > 1000
            or force
        )
        if should_flush:
            # If many draw tasks pending, wait for some to complete first
            if len(self.pending_draw_tasks) > 1000:
                self._process_completed_draw_tasks(wait_all=True)
            self._flush_completed_data(filename)

        # If force, wait for all pending draw tasks
        if force and self.pending_draw_tasks:
            self._process_completed_draw_tasks(wait_all=True)
            self._flush_completed_data(filename)
