We continued to explore the “student-teacher” paradigm. 
The SuperPoint model acted as a source for keypoint labels 
in our experiment. We employed a pretrained MobileNetV3 backbone 
and augmented it with a Feature Pyramid Network (FPN) at four 
strategic locations. Feature maps resulted from a simplistic 
aggregation strategy, involving bilinear upscaling and summation, 
followed by a Conv2D layer to reduce the feature dimensionality to 
one channel, and bicubic interpolation to upscale the output to a 
VGA resolution heatmap.

We pursued two distinct training methodologies:

* The first phase involved fine-tuning the backbone in conjunction with the FPN to cultivate a concept analogous to "keypointness," enhancing the richness of the feature maps. 

Training pipeline for the first phase:

As the backbone feature maps started to get smoothed by the keypoints, it became 
observable in the validation predictions, where yellow crosses depict 
teacher-provided ground truths. 


* The second phase—only partially explored—aimed to fine-tune the refined backbone from phase one. It incorporated label smoothing across a range of 0.1 to 0.9 and utilized a Gaussian filter to transform the binary masks into blob-like structures, preserving a maximum value of 0.9 while creating a gradient that tapers off to 0.1. The intent was to introduce both uncertainty and spatial cognition. Masks were created by aligning points between original and warped images using LightGlue. This phase was thought as a regression challenge, training concurrently on an image and its warped counterpart. A consistency loss was introduced to promote feature stability across successive frames, ensuring that predictions were robust to homography-induced perturbations. We experimented with the Huber loss for its balanced L1/L2 characteristics. 


Lastly, we also tried to keep the problem as a finetuning classification task using KL-divergence, essentially treating the image as a probability distribution. 

The consistency loss can be mathematically defined as follows: 
