"""Start the humanoid vision application."""
from config import VisionConfig
from pipeline.vision_pipeline import VisionPipeline


def main():
    VisionPipeline(VisionConfig()).run()


if __name__ == "__main__":
    main()
