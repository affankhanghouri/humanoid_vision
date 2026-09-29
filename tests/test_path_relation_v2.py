"""Deterministic v2 geometry, alignment, provenance and abstention tests."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks/path_relation'))
from core.perception_state import PerceptionEntity,PerceptionState,RoadSegObservation
from rejected_path_relation.engine_v2 import PathRelationEngineV2
from rejected_path_relation.geometry_v2 import object_probes,construct_forward_corridor_v2
from rejected_path_relation.types import PathRelationState,RoadSupportKind
from tracking.motion import MotionSnapshot,MotionStep

def entity(ts=0):return PerceptionEntity(1,'car',.9,50,42,0,0,20,16,last_observed_timestamp=ts)
def state(ts=0):return PerceptionState(round(ts*10),ts,ts,(entity(ts),),1,4,1,1)
def road(mask,ts=0):return RoadSegObservation(round(ts*10),ts,ts,1.,mask,(0,0,100,60),(100,60))

class V2Tests(unittest.TestCase):
 def test_probes_are_separated_scaled_and_bounds_aware(self):
  p=object_probes((40,20,60,40),(100,60)); self.assertEqual([x.name for x in p],['contact','below','left','right'])
  self.assertGreater(p[1].center[1],40);self.assertLess(p[2].center[0],40);self.assertGreater(p[3].center[0],60)
  self.assertFalse(object_probes((0,40,10,59),(100,60))[1].usable)
 def test_small_hole_is_closed_without_joining_remote_component(self):
  m=np.zeros((60,100),np.uint8);m[10:,20:80]=1;m[35:38,49:51]=0;m[5:10,:8]=1
  c,clean=construct_forward_corridor_v2(m,(0,0,100,60));self.assertIsNotNone(c);self.assertTrue(clean[36,50]);self.assertFalse(c.mask[7,3])
 def test_surroundings_can_support_occluded_contact(self):
  m=np.ones((60,100),np.uint8);m[47:53,47:54]=0
  e=PathRelationEngineV2();o=e.update(state(),road(m),frame_id=0,timestamp=0,frame_size=(100,60))[0]
  self.assertEqual(o.road_support_kind,RoadSupportKind.ROAD_INFERRED_FROM_SURROUNDINGS)
  self.assertNotEqual(o.state,PathRelationState.OFF_DRIVABLE)
 def test_stale_and_broken_affine_chain_abstain_without_refreshing_source(self):
  m=np.ones((60,100),np.uint8);e=PathRelationEngineV2()
  o=e.update(state(2),road(m,0),frame_id=20,timestamp=2,frame_size=(100,60))[0]
  self.assertEqual(o.state,PathRelationState.UNKNOWN);self.assertEqual(o.road_source_timestamp,0);self.assertIsNone(o.road_transform_timestamp)
  bad=MotionSnapshot((MotionStep(0,.5,np.empty((0,2)),np.empty((0,2)),None),),.5,5,(1,1))
  o=e.update(state(.5),road(m,0),frame_id=5,timestamp=.5,frame_size=(100,60),motion=bad)[0]
  self.assertEqual(o.state,PathRelationState.UNKNOWN);self.assertFalse(o.road_spatially_projected)
if __name__=='__main__':unittest.main()
