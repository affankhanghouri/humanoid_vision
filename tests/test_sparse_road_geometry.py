"""Sparse road-extent geometry tests."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from road.geometry import derive_road_geometry,conservative_corridor,project_road_geometry
from tracking.motion import MotionSnapshot,MotionStep

class RoadGeometryTests(unittest.TestCase):
 def test_connected_lower_middle_component_and_conservative_corridor(self):
  m=np.zeros((60,100),np.uint8);m[15:,15:85]=1;m[5:15,:8]=1
  g=derive_road_geometry(m,(0,0,100,60),7,1.0)
  self.assertIsNotNone(g);self.assertGreaterEqual(len(g.samples),8)
  for sample,(_,left,right) in zip(g.samples,conservative_corridor(g)):
   self.assertGreaterEqual(left,sample.left);self.assertLessEqual(right,sample.right)
 def test_missing_lower_component_abstains(self):
  m=np.zeros((60,100),np.uint8);m[:12,:20]=1
  self.assertIsNone(derive_road_geometry(m,(0,0,100,60),1,0.))
 def test_projection_uses_chain_and_broken_chain_abstains(self):
  m=np.zeros((60,100),np.uint8);m[15:,15:85]=1;g=derive_road_geometry(m,(0,0,100,60),1,0.)
  affine=np.array(((1.,0.,2.),(0.,1.,0.)));empty=np.empty((0,2))
  snap=MotionSnapshot((MotionStep(0.,.1,empty,empty,affine),),.1,2,(1.,1.))
  moved=project_road_geometry(g,snap,.1);self.assertTrue(moved.projected);self.assertAlmostEqual(moved.samples[0].center,g.samples[0].center+2)
  bad=MotionSnapshot((MotionStep(0.,.1,empty,empty,None),),.1,2,(1.,1.))
  self.assertIsNone(project_road_geometry(g,bad,.1))
if __name__=='__main__':unittest.main()
