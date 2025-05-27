# calibration_app.py

import threading
import signal
import sys
import cv2
import numpy as np
import gradio as gr
import requests
from fastapi import FastAPI
from uvicorn import Config, Server
from app_manager.patch import SilentStreamingResponse

class CalibrationApp:
    def __init__(self, camera, robot, calibrator,
                 args):

        self.camera = camera
        self.robot = robot
        self.calibrator = calibrator
        self.api_port = args.api_port
        self.ui_port = args.ui_port
        self.enable_ui = args.enable_ui

        self.stop_camera = False
        self.stream_mgr = None
        self.camera_thread = None
        self.server = None

        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/collect")
        def collect_sample():
            info = self.calibrator.collect_one_sample()
            return {info}

        @self.app.get("/calibrate")
        def calibrate():
            R, t = self.calibrator.calibrate()
            if R is None:
                return {"status": "Need more samples"}
            return {"rotation": R.tolist(), "translation": t.tolist(), "status": "Success"}

        @self.app.get("/mjpeg")
        def mjpeg_stream():
            def generate():
                try:
                    while not self.stop_camera:
                        frame = self.stream_mgr.get()
                        if frame is None:
                            continue
                        _, buffer = cv2.imencode('.jpg', frame)
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                               buffer.tobytes() + b'\r\n')
                except Exception:
                    pass
            return SilentStreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.get("/current_error")
        def current_error():
            frame, _ = self.camera.get_frame_and_depth()
            if frame is None:
                return {"error": "Failed to get frame from camera"}
            
            _, _, _, error = self.camera._extract_apriltag_pose(frame)
            if error is None:
                return {"error": "AprilTag detection failed"}
            return {"error": error.tolist()}
            
        @self.app.get("/load_history")
        def load_history():
            self.calibrator.load_history_data()
            return {"status": "History data loaded", "count": len(self.calibrator.T_base2ee)}
        
        @self.app.post("/save_data")
        def save_data():
            self.calibrator.save_data()
            return {"status": "Calibration data saved", "count": len(self.calibrator.T_base2ee)}
        
            
    def camera_loop(self):
        while not self.stop_camera:
            frame, _ = self.camera.get_frame_and_depth()
            if frame is not None:
                frame = frame[:, :, :3]
                self.stream_mgr.update(frame)

    def _cleanup(self, *args):
        print("[INFO] Cleaning up resources...")
        self.stop_camera = True
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join()
            print("[INFO] Camera thread stopped")
        try:
            self.camera.close()
            self.robot.disconnect()
            print("[INFO] Camera and robot disconnected")
        except Exception:
            pass
        if self.server:
            self.server.should_exit = True

    def _launch_gradio_ui(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Hand-Eye Calibration System")
            with gr.Row():
                collect_btn = gr.Button("Collect Sample")
                calib_btn = gr.Button("Calibrate")
                error_btn = gr.Button("Curren picture Apriltag error")
                load_data_btn = gr.Button("Load History Data")
                save_data_btn = gr.Button("Save Calibration Data")
                collect_out = gr.Textbox(label="Collect Result")
                calib_out = gr.Textbox(label="Calibration Result")
                error_out = gr.Textbox(label="Current apriltag error")
                load_data_out = gr.Textbox(label="Load status")
                save_data_out = gr.Textbox(label="Save status")

            gr.HTML(f'<img src="http://localhost:{self.api_port}/mjpeg" width="640">')

            def call_collect():
                return requests.post(f"http://localhost:{self.api_port}/collect").text

            def call_calibrate():
                return requests.get(f"http://localhost:{self.api_port}/calibrate").text
            
            def call_error():
                return requests.get(f"http://localhost:{self.api_port}/current_error").text
            
            def call_load():
                return requests.get(f"http://localhost:{self.api_port}/load_history").text
            
            def call_save():
                return requests.post(f"http://localhost:{self.api_port}/save_data").text

            collect_btn.click(fn=call_collect, outputs=collect_out)
            calib_btn.click(fn=call_calibrate, outputs=calib_out)
            error_btn.click(fn=call_error, outputs=error_out)
            load_data_btn.click(fn=call_load, outputs=load_data_out)
            save_data_btn.click(fn=call_save, outputs=save_data_out)

        return demo

    def run(self):
        from app_manager import GradioStreamManager  
        self.stream_mgr = GradioStreamManager()

        signal.signal(signal.SIGINT, self._cleanup)
        signal.signal(signal.SIGTERM, self._cleanup)

        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.start()

        if self.enable_ui:
            demo = self._launch_gradio_ui()
            threading.Thread(
                target=lambda: demo.launch(server_name="0.0.0.0", server_port=self.ui_port),
                daemon=True
            ).start()

        self.server = Server(Config(self.app, host="0.0.0.0", port=self.api_port, log_level="critical"))
        self.server.run()
