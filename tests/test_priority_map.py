import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from core.perception_state import RoadSegObservation
from rendering.priority_map import entity_priority,fresh_road_mask,render_priority_overlay

class PriorityMapTests(unittest.TestCase):
 def entity(self,kind='car',vx=0.,vy=0.,confidence=.8,misses=0):
  return SimpleNamespace(class_name=kind,velocity_x=vx,velocity_y=vy,
                         confidence=confidence,misses=misses,entity_id=1)
 def test_person_center_and_approach_raise_real_signal_score(self):
  person,_=entity_priority(self.entity('person',vy=120),(420,300,620,680),(1000,700),500)
  parked_side,_=entity_priority(self.entity('car'),(20,250,120,400),(1000,700),500)
  self.assertGreater(person,parked_side)
 def test_motion_toward_corridor_beats_motion_away(self):
  toward,_=entity_priority(self.entity('car',vx=100),(200,250,300,450),(1000,700),500)
  away,_=entity_priority(self.entity('car',vx=-100),(200,250,300,450),(1000,700),500)
  self.assertGreater(toward,away)
 def test_road_timestamp_is_checked_not_refreshed(self):
  road=RoadSegObservation(4,10.,10.05,40.,np.ones((4,8),np.uint8),(0,0,8,4),(80,40))
  self.assertIsNotNone(fresh_road_mask(road,5,11.,(80,40),1.1))
  self.assertIsNone(fresh_road_mask(road,6,11.2,(80,40),1.1))
  self.assertEqual((road.source_timestamp,road.produced_timestamp),(10.,10.05))
 def test_overlay_is_transparent_and_local(self):
  frame=np.full((200,300,3),100,np.uint8);entity=self.entity('person',vy=40)
  view=SimpleNamespace(entity=entity,box=(120,80,180,190))
  scores=render_priority_overlay(frame,[view],None,alpha=.3,scale=4)
  self.assertEqual(len(scores),1);self.assertTrue(np.any(frame!=100))
  self.assertTrue(np.all(frame[0,0]==100))

if __name__=='__main__':unittest.main()
