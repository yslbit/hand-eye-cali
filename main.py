# main.py
from app_manager import CalibrationApp
from camera_manager import Cameramanager
from sampler import HandEyeCalibrator, RobotManager
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Calibration System Launcher")
    
    parser.add_argument("--api_port", type=int, default=8000, help="FastAPI Server Port")
    parser.add_argument("--ui_port", type=int, default=7860, help="Gradio UI Port")
    parser.add_argument("--n_clusters", type=int, default=3, help="K-means Number of Clusters")
    parser.add_argument("--is_kmeans", action="store_true", help="Enable K-means clustering")
    parser.add_argument("--enable_ui", action="store_true", help="Using Gradio UI")
    parser.add_argument("--save_path", type=str, default='./samples', help="Path to Save Samples")
    parser.add_argument("--max_tag_distance", type=float, default=0.5, help="Max Tag Distance for Calibration")
    parser.add_argument("--poses_num", type=int, default=200, help="Number of Poses for Calibration")
    parser.add_argument("--tag", type=str, default='tag36h11', help="AprilTag Family")
    parser.add_argument("--tag_size", type=float, default=0.05, help="AprilTag Family Size (meters)")
    parser.add_argument("--error_threshold", type=float, default=0.5, help="AprilTag Pose Estimation Error Threshold (meters)")

    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    print(f'{args.is_kmeans},')
    
    camera = Cameramanager(args=args)
    robot = RobotManager()
    calibrator = HandEyeCalibrator(robot, 
                                   camera, 
                                   args=args
                                   )

    app = CalibrationApp(
        camera=camera,
        robot=robot,
        calibrator=calibrator,
        args=args,
    )
    app.run()
