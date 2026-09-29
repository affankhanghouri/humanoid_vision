# LaneATT study sources

The bounded lane experiment uses the official LaneATT project at revision
`2f8583ba14eccba05e6779668bc3a38bc751984a` (2021-06-28), licensed under MIT:

- Source: <https://github.com/lucastabelini/LaneATT>
- Paper: <https://openaccess.thecvf.com/content/CVPR2021/html/Tabelini_Keep_Your_Eyes_on_the_Lane_Real-Time_Attention-Guided_Lane_Detection_CVPR_2021_paper.html>
- Official pretrained archive: Google Drive file `1R638ou1AMncTCRvrkQY6I-11CPwZy23T`
- Archive member: `experiments/laneatt_r18_culane/models/model_0015.pt`
- Configuration: `cfgs/laneatt_culane_resnet18.yml`

Local model files remain ignored by Git. Artifact hashes from this run:

```text
fada2fde1d07a76fd06cce8f1e8d16283126bb4b1760b6b8d5c5088198606c2d  models/laneatt_r18_culane.pt
7ae25eaf0013b89eb0cebe5d4562569ae48521bf05b2ec1b9900cf0298a22555  models/laneatt_r18_culane.onnx
7655b10ef2272ce9ace44e8e6c30b4bca907fe4c16dc717ded7b2d5e2e373cdf  data/culane_anchors_freq.pt
```

The official CULane evaluation path resizes OpenCV BGR images to 640x360,
scales float values by 1/255, and does not apply ImageNet normalization. The
ONNX graph ends before the project's CUDA-only lane NMS. The benchmark applies
the same confidence threshold, lane-distance suppression rule, and top-four
limit on CPU.

The external PyTorch artifacts are loaded with `weights_only=True`.
