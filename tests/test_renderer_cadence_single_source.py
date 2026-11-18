"""4f: the 25->30 (or any) conform must come from ONE resolved value.

The audit named three "independent" conform sites: the in-graph cadence
fragment (`_cadence_fragment`, spliced into `_scale_pad_filter`/
`_split_filter`), zoompan's own `fps=` (`_focus_zoom_filter`,
`_freeze_pad_suffix`), and the encoder's `-r`/`-fps_mode:v cfr` (`_video_codec_args`).

Investigation (wave 8): all three already read the SAME `settings.render_fps`
field of the SAME `Settings` object threaded through `render_shot` --
`compile_shots` resolves the master cadence exactly once (`_resolve_master_cadence`
+ `model_copy`) before the shot fan-out, per `_cadence_fragment`'s own docstring
("`settings.render_fps` is already the resolved master cadence"). None of the
three sites carries an independent hardcoded literal (`_video_codec_args`
reads `settings.render_fps`; `_focus_zoom_filter`/`_freeze_pad_suffix` read
`settings.render_fps` into a local `fps` variable; `_cadence_fragment` takes
`settings.render_fps` as its target). Structurally they cannot disagree for a
GIVEN `Settings` instance -- there is nothing left to "consolidate" in this
repo. The real gap the audit was pointing at (no producer anywhere sets
`Shot.target_fps`, so `_resolve_master_cadence`'s auto selector always falls
back to `render_fps`) lives upstream of this repo, in whichever service
constructs `Shot` objects -- video_compiler_service only consumes them.

This test does NOT change behaviour. It pins the "one resolved value" property
so a future edit that hardcodes a literal `30` into any one of these three
sites (reintroducing the divergence risk the audit worried about) goes RED.
"""

from __future__ import annotations

from src.api.schemas import FocusKey, Shot
from src.config import get_settings
from src.services import cadence
from src.services.renderer import (
    _cadence_fragment,
    _focus_zoom_filter,
    _freeze_pad_suffix,
    _video_codec_args,
)


def test_encoder_cadence_fragment_and_zoompan_all_track_one_render_fps() -> None:
    settings = get_settings().model_copy(update={"render_fps": 25})

    # 1. The encoder's -r / gop args.
    encode_args = _video_codec_args(settings)
    assert "-r" in encode_args
    assert encode_args[encode_args.index("-r") + 1] == "25"
    assert "-g" in encode_args
    assert encode_args[encode_args.index("-g") + 1] == "50"  # render_fps * 2

    # 2. The in-graph cadence fragment (no declared source rate -> plain fps=).
    assert _cadence_fragment(settings) == "fps=25,setpts=PTS-STARTPTS"

    # 3. The focus-punch zoompan.
    shot = Shot(
        id="s1",
        seconds=3.0,
        video_url="http://minio/clip.mp4",
        focus_track=[FocusKey(t=0.0, bbox=(0, 0, 100, 100))],
    )
    zoompan_vf = _focus_zoom_filter("scale=1920:1080", shot, settings, duration=3.0)
    assert "fps=25" in zoompan_vf
    assert "fps=30" not in zoompan_vf

    # 4. The freeze-pad drift zoompan (narration under-run branch).
    drift = _freeze_pad_suffix(narration_driven=True, clip_dur=2.0, duration=4.0, settings=settings)
    assert "fps=25" in drift
    assert "fps=30" not in drift

    # All four sites moved together off ONE settings.render_fps change -- there
    # is no fifth site left carrying an independent literal.
    assert cadence.clamp_cadence(25) == 25
