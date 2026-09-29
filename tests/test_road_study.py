"""Boundary decoding and abstention tests for the isolated road study."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import unittest

spec = importlib.util.spec_from_file_location('road_study_geometry', Path(__file__).resolve().parents[1]/'benchmarks/road_study/geometry.py')
geometry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = geometry
spec.loader.exec_module(geometry)


def decode(curves):
    logits = np.array([[[0., 5.]] * len(curves)])
    return geometry.decode_lstr(logits, np.array([curves], dtype=float))


def pair():
    return decode([[.5, 1., 0., 0., 0., .25, 0., 0.],
                   [.5, 1., 0., 0., 0., .75, 0., 0.]])


class RoadStudyTests(unittest.TestCase):
    def test_separate_instances_and_centered_offset(self):
        lanes = pair()
        result = geometry.select_ego(lanes, np.ones((60,100),np.uint8))
        assert result['valid'] and abs(result['normalized_offset']) < 1e-8
        assert result['left_query'] != result['right_query']


    def test_missing_boundary_or_road_support_abstains(self):
        assert not geometry.select_ego(pair()[:1], np.ones((60,100),np.uint8))['valid']
        assert not geometry.select_ego(pair())['valid']
        assert not geometry.select_ego(pair(), np.zeros((60,100),np.uint8))['valid']


    def test_no_extrapolation_past_observed_extent(self):
        lanes=decode([[.5,.8,0.,0.,0.,.25,0.,0.],[.5,.8,0.,0.,0.,.75,0.,0.]])
        assert geometry.select_ego(lanes)['reason']=='missing_boundary'


    def test_invalid_numeric_output_and_poles_are_rejected(self):
        assert decode([[.5,1.,1.,.75,0.,.25,0.,0.]]) == []
        assert decode([[.5,1.,0.,0.,0.,float('nan'),0.,0.]]) == []
        with self.assertRaises(ValueError):
            geometry.decode_lstr(np.zeros((1,7,2)),np.zeros((1,7,7)))


    def test_binary_road_decode_preserves_argmax_ties(self):
        logits=np.array([[[1,2,3],[2,2,1]],[[1,3,2],[1,2,2]]])
        assert np.array_equal(logits[1]>logits[0],np.argmax(logits,axis=0))


    def test_crossing_curves_are_rejected(self):
        # Curves surround the camera at the reference row but cross further ahead.
        y=np.linspace(.5,1.,64)
        lanes=[geometry.Lane(0,.95,np.column_stack((1.1-y,y))),
               geometry.Lane(1,.95,np.column_stack((y-.1,y)))]
        assert not geometry.select_ego(lanes,np.ones((60,100),np.uint8))['valid']

    def test_selects_nearest_surrounding_instances_not_fixed_queries(self):
        lanes=decode([[.5,1.,0.,0.,0.,.1,0.,0.],
                      [.5,1.,0.,0.,0.,.8,0.,0.],
                      [.5,1.,0.,0.,0.,.3,0.,0.],
                      [.5,1.,0.,0.,0.,.7,0.,0.]])
        result=geometry.select_ego(lanes,np.ones((60,100),np.uint8))
        self.assertTrue(result['valid'])
        self.assertEqual((result['left_query'],result['right_query']),(2,3))

    def test_low_confidence_background_is_not_a_lane(self):
        logits=np.array([[[5.,0.]]])
        curves=np.array([[[.5,1.,0.,0.,0.,.3,0.,0.]]])
        self.assertEqual(geometry.decode_lstr(logits,curves),[])
