
# ML_EX3


## Notes: 

The classical algorithms are trained and run on a MacBookPro 2018. 2,2 GHz Core i7 with 16GB Ram.  
We we could we used parallel processing and training. 
The deeplearning notebook was trained with google colab using GPU runtime.


Using the pretrained VGG16 with just a simple Dense output we expericed quite some overfitting. Thus we added two Dense Layers with a Dropout in between, which reduced the overfitt. We experimented quite a lot with the layers added to the pretrained network and thus gave the best resutls for both datasets. 