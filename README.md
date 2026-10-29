# Search Token Reliability Modeling for Thermal Infrared Tracking (STRF)

Official implementation of our paper accepted by **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**.

## Abstract

Thermal infrared (TIR) tracking remains challenging due to weak textures, background heat sources, and appearance variations, which can produce unreliable search-token responses and compromise target localization. We propose Search Token Reliability Fusion (STRF), a framework that explicitly models search-token reliability and progressively refines search representations within a Transformer tracker. Specifically, TRE assesses target consistency and interference suppression, while TRM incorporates contrast and trail-continuity cues to model temporal reliability. PRR then integrates these complementary cues through gated fusion to enhance target-relevant features and suppress distracting responses. Extensive experiments on LSOTB-TIR100, LSOTB-TIR120, PTB-TIR, and LasHeR-TIR demonstrate the effectiveness and generalization ability of the proposed framework.

![Overall framework of STRF](assets/framework.png)

## Download

- **Trained models:** [Baidu Pan](https://pan.baidu.com/s/1t9qlKLA332DxVyLAvrb7nw). Extraction code: `STRF`.

- **Raw tracking results** on LSOTB-TIR100, LSOTB-TIR120, PTB-TIR, and LasHeR-TIR: [Baidu Pan](https://pan.baidu.com/s/1cbnmGu1j2SRXZhwTXF6xVw?pwd=STRF). Extraction code: `STRF`.

## Acknowledgements

Our implementation is built upon [OSTrack](https://github.com/botaoye/OSTrack). We thank the authors for making their code publicly available.
