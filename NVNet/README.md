All the required packages are in packages.py
All the data is in the data directory
    - 'cloverstray.npy' is the stray field at the service of the clover
    - 'clovermag.npy' is the magnetization of the clover

- The code should be ready to run once you have all the necessary packages and change the paths for the raw data in 
  'train.py' (LN 29) and the mask in 'model.py' (LN 7).
- You can change the amount of learnable parameters by changing the depth parameter in the model initialization (see 
  LN 8 of 'train.py')
Issues:
- I'm confused by the scaling of the model's output. When I run 'clovermag.npy' through ForwardTransform().StrayFromMag, 
  the function correctly predicts the features of the stray field, but it is off by nearly exactly a factor of 10^8
- Also, the model is reconstructing a weirder pattern than expected and I'm not sure how to fix it. I feel like it 
  would be able to succesfully reconstruct the proper clover if not for the fact that the rebuilt stray field gets
  stuck with some sign errors when compared to the source. 