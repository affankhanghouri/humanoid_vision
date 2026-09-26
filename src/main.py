"""Program entry point."""

from config import VisionConfig
from pipeline.vision_pipeline import VisionPipeline


def main():

    config = VisionConfig()

    print(
        f"Render mode: "
        f"{config.render_mode}"
    )

    pipeline = VisionPipeline(
        config
    )

    pipeline.run()


if __name__ == "__main__":
    main()