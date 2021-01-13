# Makeup Style Transfer on Low Quality Images with Weighted Multi-Scale Attention
This repository contains the networks and the evaluation code for our [ICPR Paper](http://hubertshum.com/publications/icpr2020makeup/files/icpr2020makeup.pdf). The full code and trained models are not available at this time because we continue to actively use them in our research. 

The networks file is uploaded to facilitate understanding of the proposed attention mechanism. To independently re-implement the paper, see [Augmented CycleGAN](https://github.com/aalmah/augmented_cyclegan).

The evaluation code requires masks obtained from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ). We hope this helps the community move towards a standardised comparison metric for Makeup Style Transfer.

If you use this code or find our work interesting, please cite our paper:
 ```
 @inproceedings{organisciak20makeup,
 author={Organisciak, Daniel and Ho, Edmond S. L. and Shum, Hubert P. H.},
 booktitle={Proceedings of the 2020 International Conference on Pattern Recognition},
 series={ICPR '20},
 title={Makeup Style Transfer on Low-quality Images with Weighted Multi-scale Attention},
 year={2020},
 location={Milan, Italy},
}
```
