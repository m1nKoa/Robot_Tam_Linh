#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json

class DecisionMakingNode(Node):
    def __init__(self):
        super().__init__('decision_making_node')
        
        # Publisher cho velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber cho object detection results
        self.detection_sub = self.create_subscription(
            String,
            '/object_detection_results',  # Topic từ object_detection package
            self.detection_callback,
            10
        )
        
        # Robot state - Mở rộng các states
        self.current_state = "IDLE"  
        # States: IDLE, MOVING_FORWARD, TURNING_LEFT, TURNING_RIGHT, 
        # EMERGENCY_STOP, TRAFFIC_STOP, PEDESTRIAN_STOP, STOP_LINE,
        # SLOW_DOWN, FOLLOW_CAR, AVOID_LEFT, AVOID_RIGHT,
        # HIGHWAY_ENTRY, HIGHWAY_EXIT, TURN_LEFT, TURN_RIGHT,
        # ROUNDABOUT_ENTRY, PRIORITY_ROAD
        
        self.target_object = None
        
        # Movement parameters - Thêm các tốc độ khác nhau
        self.normal_speed = 1.0
        self.slow_speed = 0.5
        self.fast_speed = 5.0
        self.angular_speed = 2.5
        self.sharp_turn_speed = 4.0
        
        # Stop timer cho các tình huống dừng tạm thời
        self.stop_timer = 0
        self.stop_duration = 30  # 3 seconds (timer runs at 10Hz)
        
        # Timer để publish commands định kỳ
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Detection history for stability
        self.last_detections = []
        self.detection_count = 0
        
        self.get_logger().info('Decision Making Node started with traffic rules support')

    def detection_callback(self, msg):
        """Callback khi nhận được kết quả detection"""
        try:
            # Parse JSON data từ object detection
            detection_data = json.loads(msg.data)
            self.process_detection(detection_data)
            self.detection_count += 1
            
            # Log every 10 detections để tránh spam
            if self.detection_count % 10 == 0:
                detected_classes = [d['class'] for d in detection_data.get('detections', [])]
                if detected_classes:
                    self.get_logger().info(f'Processing: {", ".join(set(detected_classes))}')
                    
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid JSON data received from object detection')
        except Exception as e:
            self.get_logger().error(f'Error processing detection: {e}')

    def process_detection(self, detection_data):
        """Xử lý kết quả detection và đưa ra quyết định"""
        detections = detection_data.get('detections', [])
        
        if not detections:
            # Nếu không phát hiện gì, từ từ chuyển về IDLE
            if self.current_state not in ["EMERGENCY_STOP", "TRAFFIC_STOP", "PEDESTRIAN_STOP", "STOP_LINE"]:
                self.current_state = "MOVING_FORWARD"  # Tiếp tục di chuyển
            return
        
        # Lưu detection history
        self.last_detections = detections
        
        # Phân loại objects theo mức độ ưu tiên
        critical_objects = []  # Dừng ngay
        warning_objects = []   # Cảnh báo, giảm tốc
        navigation_objects = [] # Điều hướng
        
        for detection in detections:
            obj_class = detection.get('class', '')
            confidence = detection.get('confidence', 0)
            
            # Chỉ xử lý detections có confidence cao
            if confidence < 0.6:
                continue
            
            # Objects yêu cầu dừng ngay lập tức
            if obj_class in ['stop-sign', 'traffic-red', 'pedestrian', 'stop-line']:
                critical_objects.append(detection)
            
            # Objects yêu cầu cảnh báo và giảm tốc
            elif obj_class in ['traffic-yellow', 'car', 'crosswalk-sign', 'closed-road-stand', 'no-entry-road-sign']:
                warning_objects.append(detection)
            
            # Objects cho điều hướng
            elif obj_class in ['traffic-green', 'highway-entry-sign', 'highway-exit-sign', 
                              'one-way-road-sign', 'round-about-sign', 'priority-sign',
                              'parking-sign', 'parking-spot']:
                navigation_objects.append(detection)
        
        # Xử lý theo thứ tự ưu tiên
        if critical_objects:
            self.handle_critical_objects(critical_objects)
        elif warning_objects:
            self.handle_warning_objects(warning_objects)
        elif navigation_objects:
            self.handle_navigation_objects(navigation_objects)
        else:
            # Không có object đặc biệt, tiếp tục di chuyển
            if self.current_state == "IDLE":
                self.current_state = "MOVING_FORWARD"

    def handle_critical_objects(self, objects):
        """Xử lý objects nguy hiểm - dừng ngay"""
        # Sắp xếp theo độ ưu tiên
        priority_order = ['stop-sign', 'pedestrian', 'traffic-red', 'stop-line']
        
        for priority_class in priority_order:
            for obj in objects:
                obj_class = obj.get('class', '')
                bbox = obj.get('bbox', {})
                
                if obj_class == priority_class:
                    if obj_class == 'stop-sign':
                        self.get_logger().info('STOP SIGN detected - Emergency stop!')
                        self.current_state = "EMERGENCY_STOP"
                        return
                    
                    elif obj_class == 'traffic-red':
                        self.get_logger().info('RED LIGHT detected - Stopping!')
                        self.current_state = "TRAFFIC_STOP"
                        return
                    
                    elif obj_class == 'pedestrian':
                        # Kiểm tra khoảng cách người đi bộ
                        width = bbox.get('width', 0)
                        if width > 0.1:  # Người ở gần
                            self.get_logger().info('PEDESTRIAN detected - Emergency stop!')
                            self.current_state = "PEDESTRIAN_STOP"
                            return
                    
                    elif obj_class == 'stop-line':
                        self.get_logger().info('STOP LINE detected - Stopping!')
                        self.current_state = "STOP_LINE"
                        return

    def handle_warning_objects(self, objects):
        """Xử lý objects cảnh báo - giảm tốc hoặc tránh"""
        for obj in objects:
            obj_class = obj.get('class', '')
            bbox = obj.get('bbox', {})
            center_x = bbox.get('center_x', 0.5)
            width = bbox.get('width', 0)
            
            if obj_class == 'traffic-yellow':
                self.get_logger().info('YELLOW LIGHT detected - Slowing down!')
                self.current_state = "SLOW_DOWN"
                return
            
            elif obj_class == 'car':
                # Xe ở phía trước
                if width > 0.25:  # Xe ở gần
                    self.get_logger().info('🚗 CAR ahead - Following at safe distance!')
                    self.current_state = "FOLLOW_CAR"
                    return
                else:
                    # Xe ở xa, có thể tránh
                    if center_x < 0.4:  # Xe ở bên trái
                        self.current_state = "AVOID_LEFT"
                    elif center_x > 0.6:  # Xe ở bên phải
                        self.current_state = "AVOID_RIGHT"
                    else:
                        self.current_state = "SLOW_DOWN"
                    return
            
            elif obj_class in ['crosswalk-sign', 'closed-road-stand']:
                self.get_logger().info(f'{obj_class} detected - Slowing down!')
                self.current_state = "SLOW_DOWN"
                return
                
            elif obj_class == 'no-entry-road-sign':
                self.get_logger().info('NO ENTRY detected - Stopping!')
                self.current_state = "EMERGENCY_STOP"
                return

    def handle_navigation_objects(self, objects):
        """Xử lý objects điều hướng"""
        for obj in objects:
            obj_class = obj.get('class', '')
            bbox = obj.get('bbox', {})
            center_x = bbox.get('center_x', 0.5)
            
            if obj_class == 'traffic-green':
                self.get_logger().info('GREEN LIGHT detected - Safe to proceed!')
                self.current_state = "MOVING_FORWARD"
                return
            
            elif obj_class == 'highway-entry-sign':
                self.get_logger().info('Highway entry detected - Preparing to merge!')
                self.current_state = "HIGHWAY_ENTRY"
                return
            
            elif obj_class == 'highway-exit-sign':
                self.get_logger().info('Highway exit detected!')
                self.current_state = "HIGHWAY_EXIT"
                return
            
            elif obj_class == 'one-way-road-sign':
                self.get_logger().info('One way detected!')
                self.current_state = "MOVING_FORWARD"
                return
            
            elif obj_class == 'round-about-sign':
                self.get_logger().info('Roundabout ahead - Preparing to enter!')
                self.current_state = "ROUNDABOUT_ENTRY"
                return
            
            elif obj_class == 'priority-sign':
                self.get_logger().info('Priority road ahead!')
                self.current_state = "PRIORITY_ROAD"
                return
                
            elif obj_class in ['parking-sign', 'parking-spot']:
                # Có thể thêm logic đỗ xe ở đây
                self.get_logger().info(f'{obj_class} detected!')
                self.current_state = "SLOW_DOWN"
                return

    def control_loop(self):
        """Main control loop - publish velocity commands"""
        twist = Twist()
        
        # Emergency stops - Dừng hoàn toàn
        if self.current_state in ["EMERGENCY_STOP", "TRAFFIC_STOP", "PEDESTRIAN_STOP", "STOP_LINE"]:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            # Tăng stop timer
            self.stop_timer += 1
            # Sau một thời gian, chuyển về IDLE để có thể tiếp tục
            if self.stop_timer > self.stop_duration:
                self.current_state = "IDLE"
                self.stop_timer = 0
        
        # Forward movements
        elif self.current_state == "MOVING_FORWARD":
            twist.linear.x = self.normal_speed
            twist.angular.z = 0.0
            
        elif self.current_state == "SLOW_DOWN":
            twist.linear.x = self.slow_speed
            twist.angular.z = 0.0
            
        elif self.current_state == "FOLLOW_CAR":
            twist.linear.x = self.slow_speed
            twist.angular.z = 0.0
            
        # Turning movements
        elif self.current_state == "TURNING_LEFT":
            twist.linear.x = self.slow_speed
            twist.angular.z = self.angular_speed
            
        elif self.current_state == "TURNING_RIGHT":
            twist.linear.x = self.slow_speed
            twist.angular.z = -self.angular_speed
            
        elif self.current_state == "TURN_LEFT":
            twist.linear.x = self.slow_speed
            twist.angular.z = self.sharp_turn_speed
            
        elif self.current_state == "TURN_RIGHT":
            twist.linear.x = self.slow_speed
            twist.angular.z = -self.sharp_turn_speed
            
        # Avoidance maneuvers
        elif self.current_state == "AVOID_LEFT":
            twist.linear.x = self.slow_speed
            twist.angular.z = self.angular_speed
            
        elif self.current_state == "AVOID_RIGHT":
            twist.linear.x = self.slow_speed
            twist.angular.z = -self.angular_speed
            
        # Highway maneuvers
        elif self.current_state == "HIGHWAY_ENTRY":
            twist.linear.x = self.fast_speed
            twist.angular.z = -0.3  # Slight right turn for merging
            
        elif self.current_state == "HIGHWAY_EXIT":
            twist.linear.x = self.normal_speed
            twist.angular.z = 0.3  # Slight left turn for exiting
            
        elif self.current_state == "ROUNDABOUT_ENTRY":
            twist.linear.x = self.slow_speed
            twist.angular.z = -0.5  # Right turn for roundabout
            
        elif self.current_state == "PRIORITY_ROAD":
            twist.linear.x = self.normal_speed
            twist.angular.z = 0.0
            
        else:  # IDLE
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.stop_timer = 0
        
        self.cmd_vel_pub.publish(twist)
        
        # Log current state changes
        if hasattr(self, '_last_state') and self._last_state != self.current_state:
            self.get_logger().info(f'State changed to: {self.current_state}')
        self._last_state = self.current_state

def main(args=None):
    rclpy.init(args=args)
    node = DecisionMakingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
