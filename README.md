# Search Token Reliability Modeling for Thermal Infrared Tracking (STRF)

Official implementation of our paper accepted by **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**.

## Abstract

Thermal infrared (TIR) tracking remains challenging due to weak textures, background heat sources, and appearance variations, which can produce unreliable search-token responses and compromise target localization. We propose Search Token Reliability Fusion (STRF), a framework that explicitly models search-token reliability and progressively refines search representations within a Transformer tracker. Specifically, TRE assesses target consistency and interference suppression, while TRM incorporates contrast and trail-continuity cues to model temporal reliability. PRR then integrates these complementary cues through gated fusion to enhance target-relevant features and suppress distracting responses. Extensive experiments on LSOTB-TIR100, LSOTB-TIR120, PTB-TIR, and LasHeR-TIR demonstrate the effectiveness and generalization ability of the proposed framework.

![Overall framework of STRF](assets/framework.png)

## Download

- **Trained models:** [Baidu Pan](通过网盘分享的文件：OSTrack_ep0020.pth.tar
链接: https://pan.baidu.com/s/1t9qlKLA332DxVyLAvrb7nw 提取码: STRF 
--来自百度网盘超级会员v6的分享). Extraction code: `STRF`.

- **Raw tracking results** on LSOTB-TIR100, LSOTB-TIR120, PTB-TIR, and LasHeR-TIR: [Baidu Pan](通过网盘分享的文件：STRF_raw_results.zip
链接: https://pan.baidu.com/s/1cbnmGu1j2SRXZhwTXF6xVw?pwd=STRF 提取码: STRF 
--来自百度网盘超级会员v6的分享). Extraction code: `STRF`.

## Acknowledgements

Our implementation is built upon [OSTrack](https://github.com/botaoye/OSTrack). We thank the authors for making their code publicly available.
