import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from ultralytics import YOLO
from cv_bridge import CvBridge
import cv2
import os
import json
import numpy as np

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # ========== MODEL LOADING ==========
        # Cập nhật path cho PC của bạn
        model_path = "/home/tobi/dev_ws/src/object_detection/best.pt"  # Sửa path này
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found at: {model_path}')
            self.get_logger().info('Please update the model path in the code')
            return
            
        self.yolov8_model = YOLO(model_path)
        self.class_names = self.yolov8_model.names
        self.class_names = list(self.class_names.values())
        print("Detected classes:", self.class_names)
        self.bridge = CvBridge()
        
        # ========== NETWORK OPTIMIZATION ==========
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        
        # QoS profile giống với publisher
        network_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # ========== SUBSCRIBERS ==========
        # Primary: Compressed image subscriber
        self.compressed_subscriber = self.create_subscription(
            CompressedImage,
            'camera/image_raw/compressed',  # Topic mới
            self.compressed_image_callback,
            network_qos
        )
        
        # Backup: Raw image subscriber
        self.subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            network_qos
        )
        
        # ========== PUBLISHERS ==========
        self.image_publisher = self.create_publisher(Image, 'result/image_raw', 10)
        self.detection_publisher = self.create_publisher(String, 'object_detection_results', 10)
        
        # ========== PROCESSING CONTROL ==========
        self.last_compressed_time = 0
        self.last_raw_time = 0
        self.prefer_compressed = True  # Ưu tiên compressed
        self.processing = False  # Flag để tránh overlap
        
        # ========== STATISTICS ==========
        self.total_frames = 0
        self.compressed_frames = 0
        self.raw_frames = 0
        self.last_stats_time = self.get_clock().now()
        
        print("ObjectDetectionNode Process ID: ", os.getpid())
        self.get_logger().info('Optimized Object Detection Node started')
        self.get_logger().info('Listening for both compressed and raw images')

    def compressed_image_callback(self, msg):
        """Handle compressed image - PRIMARY"""
        if self.processing:
            return  # Skip if still processing
            
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                self.last_compressed_time = self.get_clock().now().nanoseconds
                self.compressed_frames += 1
                self.process_image(cv_image, source="compressed")
            else:
                self.get_logger().warn('Failed to decode compressed image')
                
        except Exception as e:
            self.get_logger().error(f'Compressed image callback error: {e}')

    def image_callback(self, msg):
        """Handle raw image - BACKUP"""
        # Only process raw if no recent compressed image
        current_time = self.get_clock().now().nanoseconds
        time_since_compressed = (current_time - self.last_compressed_time) / 1e9
        
        if self.processing:
            return
            
        # Use raw image only if no compressed in last 0.5 seconds
        if self.prefer_compressed and time_since_compressed < 0.5:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_raw_time = current_time
            self.raw_frames += 1
            self.process_image(cv_image, source="raw")
            
        except Exception as e:
            self.get_logger().error(f'Raw image callback error: {e}')

    def process_image(self, cv_image, source="unknown"):
        """Common image processing function"""
        self.processing = True
        
        try:
            visual_image = cv_image.copy()
            
            # Get image dimensions for normalization
            image_height, image_width = cv_image.shape[:2]
            
            # ========== OBJECT DETECTION ==========
            results = self.yolov8_model.predict(cv_image, conf=0.5, show=False, verbose=False)
            
            # Prepare detection data for decision making
            detection_data = {
                'timestamp': self.get_clock().now().nanoseconds,
                'source': source,  # Add source info
                'image_width': image_width,
                'image_height': image_height,
                'detections': []
            }
            
            # ========== PROCESS DETECTIONS ==========
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes.cpu().numpy():
                        bbox = box.xyxy[0].astype(int)
                        x0, y0, x1, y1 = bbox
                        score = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.class_names[class_id]
                        
                        # Calculate normalized coordinates and dimensions
                        center_x = ((x0 + x1) / 2) / image_width  # Normalized 0-1
                        center_y = ((y0 + y1) / 2) / image_height  # Normalized 0-1
                        width = (x1 - x0) / image_width  # Normalized 0-1
                        height = (y1 - y0) / image_height  # Normalized 0-1
                        
                        # Add detection info
                        detection_info = {
                            'class': class_name,
                            'class_id': class_id,
                            'confidence': score,
                            'bbox': {
                                'x0': int(x0), 'y0': int(y0), 'x1': int(x1), 'y1': int(y1),
                                'center_x': center_x,
                                'center_y': center_y,
                                'width': width,
                                'height': height
                            }
                        }
                        detection_data['detections'].append(detection_info)
                        
                        # ========== DRAW VISUALIZATION ==========
                        color = self.get_class_color(class_name)
                        cv2.rectangle(visual_image, (x0, y0), (x1, y1), color, 2)

                        # Dynamically adjust font scale and thickness based on bbox height
                        bbox_height = y1 - y0
                        font_scale = max(0.4, min(1.0, bbox_height / 80.0))  # Better scaling
                        thickness = max(1, int(bbox_height / 40.0))
                        
                        # Display class name and confidence
                        label = f"{class_name}: {score:.2f}"
                        
                        # Add background for text
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        cv2.rectangle(visual_image, (x0, y0 - text_height - 10), (x0 + text_width, y0), color, -1)
                        cv2.putText(visual_image, label, (x0, y0 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # ========== PUBLISH RESULTS ==========
            # Publish visual result
            result_msg = self.bridge.cv2_to_imgmsg(visual_image, encoding='bgr8')
            result_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_publisher.publish(result_msg)
            
            # Publish detection data as JSON
            detection_json = json.dumps(detection_data, indent=None)  # No indentation for smaller size
            detection_msg = String()
            detection_msg.data = detection_json
            self.detection_publisher.publish(detection_msg)
            
            # ========== LOGGING & STATISTICS ==========
            self.total_frames += 1
            current_time = self.get_clock().now()
            time_diff = (current_time - self.last_stats_time).nanoseconds / 1e9
            
            if time_diff >= 10.0:  # Stats every 10 seconds
                total_fps = self.total_frames / time_diff
                compressed_fps = self.compressed_frames / time_diff
                raw_fps = self.raw_frames / time_diff
                
                self.get_logger().info(
                    f'FPS - Total: {total_fps:.1f}, Compressed: {compressed_fps:.1f}, Raw: {raw_fps:.1f}'
                )
                
                # Reset counters
                self.total_frames = 0
                self.compressed_frames = 0
                self.raw_frames = 0
                self.last_stats_time = current_time
            
            # Log detections
            if detection_data['detections']:
                detected_classes = [d['class'] for d in detection_data['detections']]
                unique_classes = list(set(detected_classes))
                self.get_logger().info(f'[{source}] Detected: {", ".join(unique_classes)}')
                
        except Exception as e:
            self.get_logger().error(f'Failed to process image from {source}: {e}')
        finally:
            self.processing = False

    def get_class_color(self, class_name):
        """Trả về màu sắc cho từng loại object"""
        color_map = {
            # Critical objects - Red
            'stop-sign': (0, 0, 255),
            'traffic-red': (0, 0, 255),
            'pedestrian': (0, 0, 255),
            'stop-line': (0, 0, 255),
            
            # Warning objects - Orange/Yellow
            'traffic-yellow': (0, 165, 255),
            'car': (0, 255, 255),
            'crosswalk-sign': (0, 165, 255),
            'closed-road-stand': (0, 165, 255),
            
            # Navigation objects - Green/Blue
            'traffic-green': (0, 255, 0),
            'highway-entry-sign': (255, 0, 0),
            'highway-exit-sign': (255, 0, 0),
            'one-way-road-sign': (255, 0, 0),
            'round-about-sign': (255, 0, 0),
            'priority-sign': (0, 255, 0),
            
            # Other objects - Default green
            'parking-sign': (0, 255, 0),
            'parking-spot': (0, 255, 0),
            'no-entry-road-sign': (0, 0, 255)
        }
        return color_map.get(class_name, (0, 255, 0))  # Default green

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
