from typing import List, Optional
import argparse
from elefant.config import load_config, ConfigBase, WandbConfig
import os
import datetime
import logging
import wandb
import pydantic
import tempfile
import concurrent
import dateutil.parser
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback
from elefant.data.rescale.rescale import rescale_local_video


class RescaleConfig(ConfigBase):
    wandb: WandbConfig = pydantic.Field(default=WandbConfig())
    prefix: str
    frame_height: int
    frame_width: int
    n_threads: int = 1

    # If fps is set, recode the video to the given fps.
    fps: Optional[int] = None
    encode_type: Optional[str] = None
    encode_color_space: str = "yuv"
    quality_factor: Optional[str] = None
    use_fast_decode: Optional[bool] = None
    preset: Optional[str] = None
    probability_of_nvidia_encoding: float = 0.9


class Rescaler:
    def __init__(self, config: RescaleConfig):
        self.config = config
        self._data_dir = os.path.join(os.getcwd(), self.config.prefix)
        logging.info(f"Using data dir: {self._data_dir}")

        if self.config.n_threads > 1:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.n_threads)

        if self.config.wandb.enabled:
            wandb.init(
                project=self.config.wandb.project,
                name=self.config.wandb.exp_name,
                tags=self.config.wandb.tags,
                config=self.config.model_dump(),
            )

        self._n_rescaled = 0
        self._n_skipped = 0
        self._n_errors = 0
        # Make a mutex for the wandb logging.
        self._wandb_lock = threading.Lock()

    def update_wandb_stats(self, skipped: bool = False, error: bool = False):
        with self._wandb_lock:
            if skipped:
                self._n_skipped += 1
            elif error:
                self._n_errors += 1
            else:
                self._n_rescaled += 1
            wandb.log(
                {
                    "n_rescaled": self._n_rescaled,
                    "n_skipped": self._n_skipped,
                    "n_errors": self._n_errors,
                }
            )

    def run(self):
        all_videos = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(self._data_dir)
            for f in files
            if f.endswith(".mp4")
        )
        videos = [v for v in all_videos if "video.mp4" in v]

        futures = []
        for video_path in videos:
            if self.config.n_threads == 1:
                futures.append(self._check_and_rescale_video(video_path, all_videos))
            else:
                futures.append(
                    self._thread_pool.submit(
                        self._check_and_rescale_video, video_path, all_videos
                    )
                )

        if self.config.n_threads > 1:
            # Wait for all futures to complete and raise any exceptions
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def _check_and_rescale_video(self, video_path: str, all_videos: List[str]):
        # First check if the video is already rescaled.
        output_file = f"{self.config.frame_height}x{self.config.frame_width}.mp4"
        rescaled_video_path = os.path.join(os.path.dirname(video_path), output_file)
        try:
            if rescaled_video_path in all_videos:
                logging.info(f"Video {video_path} already rescaled.")
                self.update_wandb_stats(skipped=True)
            else:
                self._recode_video(video_path, rescaled_video_path)
                self.update_wandb_stats(skipped=False)
        except Exception as e:
            logging.error(f"Error rescaling video {video_path}: {e}")
            traceback.print_exc()
            self.update_wandb_stats(error=True)

    def _recode_video(self, video_path: str, rescaled_video_path: str):
        rescale_local_video(
            video_path,
            self.config.frame_height,
            self.config.frame_width,
            output_path=rescaled_video_path,
            fps=self.config.fps,
            rescale_config=self.config,
            probability_of_nvidia_encoding=self.config.probability_of_nvidia_encoding,
        )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(filename)s:%(lineno)d %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config, RescaleConfig)
    Rescaler(config).run()


if __name__ == "__main__":
    wandb.init(project="open-p2p", name="rescale")
    main()
