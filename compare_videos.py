import argparse
import os
from glob import glob
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, clips_array

def auto_find_mp4(folder: str) -> str:
    """Return newest non-empty .mp4 in folder."""
    files = [p for p in glob(os.path.join(folder, "*.mp4")) if os.path.getsize(p) > 0]
    if not files:
        raise FileNotFoundError(f"No .mp4 found in {folder}")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

# def load_with_title(path: str, label: str, target_h: int = 240, max_secs: int = 60, font: str | None = None):
#     """Load video, trim to <= max_secs, resize, and overlay a title bar."""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing video: {path}")

#     clip = VideoFileClip(path)

#     # Clamp to duration
#     end_t = min(max_secs, clip.duration or max_secs) if max_secs is not None else clip.duration
#     if hasattr(clip, "subclip"):        # MoviePy 1.x
#         clip = clip.subclip(0, end_t)
#     elif hasattr(clip, "subclipped"):   # MoviePy 2.x
#         clip = clip.subclipped(0, end_t)

#     # Resize (MoviePy 2.x uses 'resized')
#     if hasattr(clip, "resize"):
#         clip = clip.resize(height=target_h)
#     else:
#         clip = clip.resized(height=target_h)

#     w, _ = clip.size

#     # Title text (MoviePy 2.x: 'text=' and 'font_size=')
#     # Use an explicit font path if provided; otherwise let Pillow choose a default.
#     text_kwargs = dict(text=label, font_size=24, color="white")
#     if font:  # you can pass a .ttf/.otf path or installed font name
#         text_kwargs["font"] = font

#     title_txt = TextClip(**text_kwargs).with_duration(clip.duration)

#     # Black bar background for readability
#     title_bar = ColorClip(size=(w, 30), color=(0, 0, 0)).with_duration(clip.duration)

#     # Position: bar at top, text centered on the bar
#     comp = CompositeVideoClip([
#         clip,
#         title_bar.set_position(("center", "top")),
#         title_txt.set_position(("center", "top"))
#     ])
#     return comp

# def load_with_title(path: str, label: str, target_h: int = 240, max_secs: int = 60, font: str | None = None):
#     """Load video, trim to <= max_secs, resize, and overlay a title bar."""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing video: {path}")

#     clip = VideoFileClip(path)

#     # Clamp to duration
#     end_t = min(max_secs, clip.duration or max_secs) if max_secs is not None else clip.duration
#     if hasattr(clip, "subclip"):        # MoviePy 1.x
#         clip = clip.subclip(0, end_t)
#     elif hasattr(clip, "subclipped"):   # MoviePy 2.x
#         clip = clip.subclipped(0, end_t)

#     # Resize (MoviePy 2.x uses 'resized')
#     if hasattr(clip, "resize"):
#         clip = clip.resize(height=target_h)
#     else:
#         clip = clip.resized(height=target_h)

#     w, _ = clip.size

#     # Title text (MoviePy 2.x: 'text=' and 'font_size=')
#     # Use an explicit font path if provided; otherwise let Pillow choose a default.
#     text_kwargs = dict(text=label, font_size=24, color="white")
#     if font:  # you can pass a .ttf/.otf path or installed font name
#         text_kwargs["font"] = font

#     title_txt = TextClip(**text_kwargs).with_duration(clip.duration)

#     # Black bar background for readability
#     title_bar = ColorClip(size=(w, 30), color=(0, 0, 0)).with_duration(clip.duration)

#     # Position: bar at top, text centered on the bar
#     comp = CompositeVideoClip([
#         clip,
#         title_bar.with_position(("center", "top")),  # Changed from set_position
#         title_txt.with_position(("center", "top"))   # Changed from set_position
#     ])
#     return comp

def load_with_title(path: str, label: str, target_h: int = 240, max_secs: int = 60, font: str | None = None):
    """Load video, trim to <= max_secs, resize, and overlay a title bar."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing video: {path}")

    clip = VideoFileClip(path)

    # Clamp to duration
    end_t = min(max_secs, clip.duration or max_secs) if max_secs is not None else clip.duration
    if hasattr(clip, "subclip"):        # MoviePy 1.x
        clip = clip.subclip(0, end_t)
    elif hasattr(clip, "subclipped"):   # MoviePy 2.x
        clip = clip.subclipped(0, end_t)

    # Resize (MoviePy 2.x uses 'resized')
    if hasattr(clip, "resize"):
        clip = clip.resize(height=target_h)
    else:
        clip = clip.resized(height=target_h)

    w, _ = clip.size

    # Title text (MoviePy 2.x: 'text=' and 'font_size=')
    # Use an explicit font path if provided; otherwise let Pillow choose a default.
    text_kwargs = dict(text=label, font_size=24, color="white", size=(w, None))  # Let text width adapt to content
    if font:  # you can pass a .ttf/.otf path or installed font name
        text_kwargs["font"] = font

    title_txt = TextClip(**text_kwargs).with_duration(clip.duration)

    # Black bar background for readability, increased height for longer text
    title_bar = ColorClip(size=(w, 50), color=(0, 0, 0)).with_duration(clip.duration)

    # Position: bar at top, text centered on the bar with a slight offset
    comp = CompositeVideoClip([
        clip,
        title_bar.with_position(("center", "top")),
        title_txt.with_position(("center", 5))  # Offset text 5 pixels down from the top
    ])
    return comp



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", default="runs_cpu_baseline/video", help="CPU baseline folder or file")
    parser.add_argument("--gpu_dqn", default="runs_gpu_dqn/video", help="GPU DQN folder or file")
    parser.add_argument("--gpu_ddqn", default="runs_gpu_ddqn/video", help="GPU DDQN folder or file")
    parser.add_argument("--out", default="comparison.mp4", help="Output video filename")
    parser.add_argument("--max_secs", type=int, default=60, help="Max seconds per clip")
    parser.add_argument("--height", type=int, default=240, help="Height (px) of each panel")
    parser.add_argument("--layout", choices=["vertical", "horizontal"], default="vertical",
                        help="Stack panels vertically or side-by-side")
    parser.add_argument("--font", default=None, help="Optional font name or path to .ttf/.otf for titles")
    args = parser.parse_args()

    # Accept folders or explicit files
    cpu_path = args.cpu if args.cpu.lower().endswith(".mp4") else auto_find_mp4(args.cpu)
    dqn_path = args.gpu_dqn if args.gpu_dqn.lower().endswith(".mp4") else auto_find_mp4(args.gpu_dqn)
    ddqn_path = args.gpu_ddqn if args.gpu_ddqn.lower().endswith(".mp4") else auto_find_mp4(args.gpu_ddqn)

    panels = [
        load_with_title(cpu_path, "CPU Baseline", target_h=args.height, max_secs=args.max_secs, font=args.font),
        load_with_title(dqn_path, "GPU DQN", target_h=args.height, max_secs=args.max_secs, font=args.font),
        load_with_title(ddqn_path, "GPU DDQN", target_h=args.height, max_secs=args.max_secs, font=args.font),
    ]

    # Layout
    if args.layout == "vertical":
        final = clips_array([[panels[0]], [panels[1]], [panels[2]]])
    else:
        final = clips_array([panels])

    # Encode (constant fps; no audio)
    final.write_videofile(args.out, codec="libx264", audio=False, fps=30, bitrate="3000k")

if __name__ == "__main__":
    main()
