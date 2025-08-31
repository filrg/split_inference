from src.Tracker import Tracker
import yaml
import pika
import os

if __name__ == "__main__":
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        tracker_app = Tracker(config)
        tracker_app.run()

    except FileNotFoundError:
        print("Error: 'config.yaml' not found. Please run this script from the project root.")
    except Exception as e:
        print(f"Failed to start Tracker: {e}")