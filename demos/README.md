# Recorded demos

```bash
python3 -m http.server 8000 --directory demos
```

Open http://localhost:8000. This static gallery makes no model requests and
requires no third-party assets at runtime. Click a video to play it using the
browser's native controls.

| File | Recording |
| --- | --- |
| `media/vizdoom.mp4` | Jev / NanoJev / GemmaJev, ViZDoom aiming and firing |
| `media/maze.mp4` | Jev / NanoJev / GemmaJev, one 50×50 maze |
| `media/three-mazes.mp4` | Movement-trained GemmaJev, three fixed 50×50 mazes |

The comparison clips use the initial two-game checkpoint. The three-maze clip
uses the movement-trained checkpoint and is accelerated to about 14 seconds.
These three maps are included in its training. Gemma chooses at junctions; code
tracks visits, masks blocked moves and handles backtracking. See the
[data and matched evaluation](../docs/navigation.md).
All clips are replays. Playback speed does not measure inference speed.
The first two GIFs are reduced-resolution previews of the corresponding MP4s.
Checksums and checkpoint labels are recorded in [media/sources.json](media/sources.json).

## Credits

The comparison recordings include public Jev and NanoJev trajectories from
[TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev), together with our
trained Gemma trajectories. NanoJev source and replay code use the MIT license;
see [NANOJEV-LICENSE](NANOJEV-LICENSE).

Game imagery comes from the ViZDoom Basic scenario; ViZDoom and its bundled
scenario assets retain their respective licenses. Videos contain rendered text,
not redistributable font files. Apache-2.0 applies to this project's code and
does not replace third-party asset terms.

To run a model and create your own recordings, follow [the demo guide](../docs/demos.md).
