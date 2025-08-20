import time
import statistics
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages performance metrics"""
    
    def __init__(self):
        self.frame_metrics: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.total_frames = 0
        self.successful_detections = 0
        
    def add_frame_metrics(self, frame_id: str, capture_ts: int, recv_ts: int, 
                         inference_ts: int, detections_count: int):
        """Add metrics for a processed frame"""
        try:
            current_ts = int(time.time() * 1000)
            
            # Calculate latencies
            network_latency = recv_ts - capture_ts
            server_latency = inference_ts - recv_ts
            end_to_end_latency = current_ts - capture_ts
            
            metrics = {
                "frame_id": frame_id,
                "capture_ts": capture_ts,
                "recv_ts": recv_ts,
                "inference_ts": inference_ts,
                "display_ts": current_ts,
                "network_latency": network_latency,
                "server_latency": server_latency,
                "end_to_end_latency": end_to_end_latency,
                "detections_count": detections_count
            }
            
            self.frame_metrics.append(metrics)
            self.total_frames += 1
            
            if detections_count > 0:
                self.successful_detections += 1
                
            # Keep only last 1000 frames to avoid memory issues
            if len(self.frame_metrics) > 1000:
                self.frame_metrics = self.frame_metrics[-1000:]
                
        except Exception as e:
            logger.error(f"Error adding frame metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            if not self.frame_metrics:
                return {
                    "total_frames": 0,
                    "processed_fps": 0,
                    "median_e2e_latency": 0,
                    "p95_e2e_latency": 0,
                    "median_network_latency": 0,
                    "median_server_latency": 0,
                    "detection_rate": 0
                }
            
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            processed_fps = self.total_frames / elapsed_time if elapsed_time > 0 else 0
            
            # Extract latency values
            e2e_latencies = [m["end_to_end_latency"] for m in self.frame_metrics]
            network_latencies = [m["network_latency"] for m in self.frame_metrics]
            server_latencies = [m["server_latency"] for m in self.frame_metrics]
            
            # Calculate statistics
            median_e2e = statistics.median(e2e_latencies) if e2e_latencies else 0
            p95_e2e = self._percentile(e2e_latencies, 95) if e2e_latencies else 0
            median_network = statistics.median(network_latencies) if network_latencies else 0
            median_server = statistics.median(server_latencies) if server_latencies else 0
            
            detection_rate = (self.successful_detections / self.total_frames) if self.total_frames > 0 else 0
            
            return {
                "total_frames": self.total_frames,
                "processed_fps": round(processed_fps, 2),
                "median_e2e_latency": round(median_e2e, 2),
                "p95_e2e_latency": round(p95_e2e, 2),
                "median_network_latency": round(median_network, 2),
                "median_server_latency": round(median_server, 2),
                "detection_rate": round(detection_rate, 3),
                "elapsed_time": round(elapsed_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export comprehensive metrics for benchmarking"""
        try:
            current_metrics = self.get_metrics()
            
            # Add additional details for export
            export_data = {
                "timestamp": int(time.time()),
                "summary": current_metrics,
                "detailed_frames": self.frame_metrics[-100:],  # Last 100 frames
                "configuration": {
                    "total_frames_processed": self.total_frames,
                    "successful_detections": self.successful_detections,
                    "collection_duration": time.time() - self.start_time
                }
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return {}
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        self.frame_metrics.clear()
        self.start_time = time.time()
        self.total_frames = 0
        self.successful_detections = 0
        logger.info("Metrics reset")
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a dataset"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        
        if index >= len(sorted_data):
            return sorted_data[-1]
        
        return sorted_data[index]
