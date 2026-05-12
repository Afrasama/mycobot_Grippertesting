#!/usr/bin/env python3
"""
Autonomous object detection and intelligent goal selection system
"""
import pybullet as p
import numpy as np
from typing import List, Tuple, Optional, Dict
from .segmentation import get_overhead_camera_image, compute_body_centroid


class AutonomousNavigator:
    """Intelligent object detection and placement system"""
    
    def __init__(self, workspace_bounds: Dict[str, Tuple[float, float]] = None):
        """
        Initialize autonomous navigator
        
        Args:
            workspace_bounds: Dictionary with 'x', 'y' bounds for valid placement areas
        """
        self.workspace_bounds = workspace_bounds or {
            'x': (0.1, 0.6),   # X-axis workspace limits
            'y': (-0.4, 0.4),  # Y-axis workspace limits
            'z': (0.02, 0.3)   # Z-axis placement height range
        }
        self.placement_history = []
        self.occupied_zones = []
    
    def detect_all_objects(self, exclude_body_ids: List[int] = None) -> List[Dict]:
        """
        Detect all objects in the workspace using overhead camera
        
        Args:
            exclude_body_ids: Body IDs to exclude from detection (like robot, gripper)
            
        Returns:
            List of detected objects with their properties
        """
        exclude_body_ids = exclude_body_ids or []
        
        # Get camera view
        rgb, depth, seg_mask = get_overhead_camera_image(width=640, height=480)
        
        # Find all unique body IDs in segmentation mask
        unique_body_ids = np.unique(seg_mask)
        detected_objects = []
        
        for body_id in unique_body_ids:
            if body_id <= 0 or body_id in exclude_body_ids:
                continue
                
            # Get object centroid
            centroid = compute_body_centroid(seg_mask, body_id)
            if centroid is None:
                continue
            
            # Get object info from PyBullet
            try:
                pos, orn = p.getBasePositionAndOrientation(body_id)
                
                # Get object type from visual shape data
                visual_data = p.getVisualShapeData(body_id)
                object_type = self._classify_object(visual_data, pos)
                
                detected_objects.append({
                    'body_id': body_id,
                    'position': np.array(pos),
                    'orientation': np.array(orn),
                    'centroid': centroid,
                    'type': object_type,
                    'size': self._estimate_object_size(visual_data),
                    'graspable': self._is_graspable(object_type, pos)
                })
                
            except Exception as e:
                print(f"Error getting info for body {body_id}: {e}")
                continue
        
        return detected_objects
    
    def _classify_object(self, visual_data: list, position: np.ndarray) -> str:
        """Classify object type based on visual data and position"""
        if not visual_data:
            return "unknown"
        
        # Simple classification based on visual shape properties
        _, _, _, _, _, _, visual_shape_type, dimensions = visual_data[0]
        
        if visual_shape_type == p.GEOM_BOX:
            if np.allclose(dimensions, [0.02, 0.02, 0.01], atol=0.005):
                return "cube_small"
            else:
                return "box"
        elif visual_shape_type == p.GEOM_SPHERE:
            return "sphere"
        elif visual_shape_type == p.GEOM_CYLINDER:
            return "cylinder"
        else:
            return "unknown"
    
    def _estimate_object_size(self, visual_data: list) -> Dict[str, float]:
        """Estimate object dimensions from visual data"""
        if not visual_data:
            return {'width': 0.0, 'length': 0.0, 'height': 0.0}
        
        _, _, _, _, _, _, visual_shape_type, dimensions = visual_data[0]
        
        if visual_shape_type == p.GEOM_BOX:
            # Box dimensions are half-extents, so multiply by 2
            return {
                'width': dimensions[0] * 2,
                'length': dimensions[1] * 2,
                'height': dimensions[2] * 2
            }
        else:
            return {'width': 0.05, 'length': 0.05, 'height': 0.05}  # Default estimate
    
    def _is_graspable(self, object_type: str, position: np.ndarray) -> bool:
        """Determine if object is graspable based on type and position"""
        graspable_types = ["cube_small", "box", "sphere", "cylinder"]
        
        # Check if object is within robot reach
        if not (self.workspace_bounds['x'][0] <= position[0] <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= position[1] <= self.workspace_bounds['y'][1]):
            return False
        
        return object_type in graspable_types
    
    def select_target_object(self, detected_objects: List[Dict]) -> Optional[Dict]:
        """
        Intelligently select the best object to pick
        
        Args:
            detected_objects: List of detected objects
            
        Returns:
            Selected object or None if no suitable object found
        """
        graspable_objects = [obj for obj in detected_objects if obj['graspable']]
        
        if not graspable_objects:
            return None
        
        # Scoring system for object selection
        best_object = None
        best_score = -1
        
        for obj in graspable_objects:
            score = self._score_object(obj)
            if score > best_score:
                best_score = score
                best_object = obj
        
        return best_object
    
    def _score_object(self, obj: Dict) -> float:
        """
        Score an object for selection based on multiple criteria
        
        Args:
            obj: Object dictionary
            
        Returns:
            Selection score (higher is better)
        """
        score = 0.0
        
        # Prefer smaller objects (easier to grasp)
        size_score = 1.0 / (obj['size']['width'] + obj['size']['length'] + 0.01)
        score += size_score * 10
        
        # Prefer objects closer to robot's neutral position
        neutral_pos = np.array([0.35, 0.0])
        obj_2d = obj['position'][:2]
        distance = np.linalg.norm(obj_2d - neutral_pos)
        distance_score = 1.0 / (distance + 0.1)
        score += distance_score * 5
        
        # Prefer certain object types
        type_preferences = {
            "cube_small": 10,
            "box": 8,
            "sphere": 6,
            "cylinder": 4
        }
        score += type_preferences.get(obj['type'], 0)
        
        # Avoid recently placed areas
        for placed_pos in self.placement_history[-5:]:  # Check last 5 placements
            if np.linalg.norm(obj['position'][:2] - placed_pos[:2]) < 0.1:
                score -= 5
        
        return score
    
    def find_optimal_placement_location(self, 
                                      target_object: Dict, 
                                      detected_objects: List[Dict],
                                      task_context: str = "organize") -> np.ndarray:
        """
        Find optimal placement location based on task context and environment
        
        Args:
            target_object: Object being moved
            detected_objects: All detected objects
            task_context: Type of task ("organize", "sort", "clear", etc.)
            
        Returns:
            Optimal placement position as numpy array
        """
        if task_context == "organize":
            return self._find_organizing_placement(target_object, detected_objects)
        elif task_context == "sort":
            return self._find_sorting_placement(target_object, detected_objects)
        elif task_context == "clear":
            return self._find_clearing_placement(target_object, detected_objects)
        else:
            return self._find_default_placement(target_object, detected_objects)
    
    def _find_organizing_placement(self, target_object: Dict, detected_objects: Dict) -> np.ndarray:
        """Find placement for organizing objects neatly"""
        # Create a grid-based organization
        grid_size = 0.08
        start_x = self.workspace_bounds['x'][0] + 0.1
        start_y = self.workspace_bounds['y'][0] + 0.1
        
        # Find next available grid position
        for i in range(10):  # Try up to 10 grid positions
            for j in range(8):
                test_pos = np.array([
                    start_x + i * grid_size,
                    start_y + j * grid_size,
                    self.workspace_bounds['z'][0]
                ])
                
                if self._is_position_clear(test_pos, detected_objects, target_object):
                    return test_pos
        
        # Fallback to default placement
        return self._find_default_placement(target_object, detected_objects)
    
    def _find_sorting_placement(self, target_object: Dict, detected_objects: List[Dict]) -> np.ndarray:
        """Find placement for sorting by object type"""
        # Group similar objects together
        similar_objects = [obj for obj in detected_objects 
                         if obj['type'] == target_object['type'] and 
                         obj['body_id'] != target_object['body_id']]
        
        if similar_objects:
            # Place near similar objects
            avg_pos = np.mean([obj['position'] for obj in similar_objects], axis=0)
            offset = np.array([0.1, 0.0, 0.0])  # Offset to avoid collision
            test_pos = avg_pos + offset
            
            if self._is_position_clear(test_pos, detected_objects, target_object):
                return test_pos
        
        return self._find_default_placement(target_object, detected_objects)
    
    def _find_clearing_placement(self, target_object: Dict, detected_objects: List[Dict]) -> np.ndarray:
        """Find placement for clearing workspace (move to edges)"""
        # Move objects to the edge of workspace
        edge_y = self.workspace_bounds['y'][1] - 0.1
        
        # Find a clear spot along the edge
        for x in np.linspace(self.workspace_bounds['x'][0] + 0.05, 
                           self.workspace_bounds['x'][1] - 0.05, 10):
            test_pos = np.array([x, edge_y, self.workspace_bounds['z'][0]])
            
            if self._is_position_clear(test_pos, detected_objects, target_object):
                return test_pos
        
        return self._find_default_placement(target_object, detected_objects)
    
    def _find_default_placement(self, target_object: Dict, detected_objects: List[Dict]) -> np.ndarray:
        """Find default placement location using spatial reasoning"""
        # Use LLM-influenced intelligent placement
        best_pos = None
        best_score = -1
        
        # Sample potential positions
        for _ in range(50):  # Try 50 random positions
            test_pos = np.array([
                np.random.uniform(self.workspace_bounds['x'][0] + 0.05, 
                                self.workspace_bounds['x'][1] - 0.05),
                np.random.uniform(self.workspace_bounds['y'][0] + 0.05, 
                                self.workspace_bounds['y'][1] - 0.05),
                self.workspace_bounds['z'][0]
            ])
            
            if self._is_position_clear(test_pos, detected_objects, target_object):
                score = self._score_placement_position(test_pos, target_object, detected_objects)
                if score > best_score:
                    best_score = score
                    best_pos = test_pos
        
        return best_pos if best_pos is not None else np.array([0.45, -0.15, 0.02])
    
    def _is_position_clear(self, position: np.ndarray, 
                          detected_objects: List[Dict], 
                          target_object: Dict, 
                          clearance: float = 0.08) -> bool:
        """Check if position is clear of other objects"""
        for obj in detected_objects:
            if obj['body_id'] == target_object['body_id']:
                continue
            
            distance = np.linalg.norm(obj['position'][:2] - position[:2])
            if distance < clearance:
                return False
        
        return True
    
    def _score_placement_position(self, position: np.ndarray, 
                                target_object: Dict, 
                                detected_objects: List[Dict]) -> float:
        """Score a placement position"""
        score = 0.0
        
        # Prefer positions that maintain good workspace organization
        # Distance from other objects (not too close, not too far)
        min_distances = []
        for obj in detected_objects:
            if obj['body_id'] != target_object['body_id']:
                dist = np.linalg.norm(obj['position'][:2] - position[:2])
                min_distances.append(dist)
        
        if min_distances:
            avg_distance = np.mean(min_distances)
            # Optimal distance around 0.15m
            distance_score = 1.0 / (abs(avg_distance - 0.15) + 0.01)
            score += distance_score * 10
        
        # Prefer positions that create organized patterns
        center = np.array([0.35, 0.0])
        dist_from_center = np.linalg.norm(position[:2] - center)
        center_score = 1.0 / (dist_from_center + 0.01)
        score += center_score * 2
        
        return score
    
    def update_placement_history(self, position: np.ndarray):
        """Update history of placed objects"""
        self.placement_history.append(position)
        # Keep only recent history
        if len(self.placement_history) > 20:
            self.placement_history = self.placement_history[-20:]
