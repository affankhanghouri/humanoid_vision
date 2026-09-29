import sys,unittest
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parents[1]/'benchmarks/custom_ego_lane';sys.path.insert(0,str(HERE))
from data import flip_labels,interpolate_boundary
from losses import ego_lane_loss
from model import ROW_ANCHORS,TinyEgoLaneNet,parameter_count
from build_dataset import sampling_stride
class CustomEgoLaneTests(unittest.TestCase):
 def test_sampling_stride_covers_full_clip_under_cap(self):
  self.assertEqual(sampling_stride(3281,30.0,.1,180),19)
  self.assertLessEqual((3281-1)//19+1,180)
 def test_model_is_small_and_structured(self):
  model=TinyEgoLaneNet(False).eval()
  with torch.no_grad():x,v,s=model(torch.zeros(1,3,192,320))
  self.assertEqual(tuple(x.shape),(1,24,2));self.assertEqual(tuple(v.shape),(1,24,2));self.assertEqual(tuple(s.shape),(1,3));self.assertLess(parameter_count(model),1_100_000)
 def test_horizontal_flip_swaps_sides_and_coordinates(self):
  x=np.tile(np.array([[.2,.8]],np.float32),(ROW_ANCHORS,1));v=np.tile(np.array([[1,0]],np.float32),(ROW_ANCHORS,1));fx,fv=flip_labels(x,v)
  np.testing.assert_allclose(fx,np.tile([[.2,.8]],(ROW_ANCHORS,1)));np.testing.assert_array_equal(fv,np.tile([[0,1]],(ROW_ANCHORS,1)))
 def test_invalid_frames_do_not_have_coordinate_loss(self):
  pred=(torch.rand(2,24,2),torch.zeros(2,24,2),torch.zeros(2,3));coords=torch.rand(2,24,2);visible=torch.zeros(2,24,2);state=torch.ones(2,dtype=torch.long)
  loss=ego_lane_loss(pred,coords,visible,state);self.assertEqual(float(loss['coordinate']),0.0)
 def test_boundary_interpolation_only_marks_supported_y(self):
  x,v=interpolate_boundary({'status':'visible','points':[[100,360],[200,648]]},1000,720)
  self.assertTrue((v>0).any());self.assertTrue((v==0).any());self.assertTrue(((x>=0)&(x<=1)).all())
if __name__=='__main__':unittest.main()
