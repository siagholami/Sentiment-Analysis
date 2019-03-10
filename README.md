<html>
<head><meta content="text/html; charset=UTF-8" http-equiv="content-type"></head>
<h1 class="c11" id="h.740y7ryjroz9"><span class="c16">Sentiment Analysis</span></h1><h2 class="c12" id="h.566d222m5pfp"><span class="c0">Use Case</span></h2><p class="c6"><span class="c4">The project aims to classify the sentiment of input text into 5 categories: very bad(1), bad(2), neutral(3), good(4), and very good(5)</span></p><p class="c5"><span class="c4"></span></p><p class="c6"><span class="c4">Input (X): Text</span></p><p class="c6"><span class="c4">Output (Y): Sentiment Category (1-5)</span></p><h2 class="c12" id="h.wjwvcphm20tn"><span class="c0">Dataset </span></h2><p class="c6"><span class="c8"><a class="c9" href="https://www.google.com/url?q=https://www.yelp.com/dataset&amp;sa=D&amp;ust=1552261754867000">Yelp Dataset</a></span><span>&nbsp;which is</span><span>&nbsp;5,996,996 Yelp reviews, labeled by sentiment (1-5) in JSON format</span><span class="c4">. </span></p><h2 class="c12" id="h.s66jzo5amq9b"><span>Preprocessing</span><span class="c0">&nbsp;</span></h2><p class="c6"><span class="c4">Preprocess the dataset in the following order: </span></p><ul class="c3 lst-kix_b2lnqq41qp75-0 start"><li class="c2"><span class="c4">Lowercase all words</span></li><li class="c2"><span class="c4">I&rsquo;m > I am </span></li><li class="c2"><span class="c4">McDonald&#39;s &gt; McDonald</span></li><li class="c2"><span class="c4">Only keep words with known GloVe embeddings. </span></li></ul><p class="c5"><span class="c4"></span></p><p class="c6"><span class="c4">Since we already have pre-trained GloVe vectors, I did not stem the words; let&rsquo;s use all the information we have. </span></p><h2 class="c12" id="h.y155vda2kv9c"><span class="c0">Data ETL</span></h2><p class="c6"><span class="c4">JSON to processed and indexed HDF5 format to be used by ML models. Data stored in AWS S3. </span></p><p class="c5"><span class="c4"></span></p><p class="c1"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 431.00px; height: 81.00px;"><img alt="" src="images/image1.png" style="width: 431.00px; height: 81.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c1"><span class="c4">Fig 1. Sentiment Analysis Data Pipeline</span></p><p class="c1 c17"><span class="c4"></span></p><p class="c6"><span class="c4">All ETL jobs are done in Python. </span></p><h2 class="c12" id="h.gc4ez7fwl6yc"><span class="c0">ML Models</span></h2><p class="c6"><span class="c4">Evaluated the following models for this project: </span></p><h3 class="c7" id="h.erjj4zp3mm4e"><span class="c14">Global Vectors for Word Representations (GloVe)</span></h3><p class="c6"><span>The model is based on Stanford&rsquo;s </span><span class="c8"><a class="c9" href="https://www.google.com/url?q=https://nlp.stanford.edu/projects/glove/&amp;sa=D&amp;ust=1552261754869000">GloVe project</a></span><span>&nbsp;[1]. I added a dense layer on top of pre-trained GloVe layer to classify the text. </span><span class="c4">&nbsp;</span></p><h2 class="c13" id="h.lbfz9t3zg2ms"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 361.00px; height: 198.00px;"><img alt="" src="images/image6.png" style="width: 361.00px; height: 198.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></h2><p class="c1"><span>Fig 2. GloVe average model architecture</span></p><p class="c5"><span class="c4"></span></p><p class="c6"><span>Implemented in TensorFlow. </span></p><h3 class="c7" id="h.vov4jwvi8oqd"><span class="c14">GloVe + Long Short-term Memory (LSTM) </span></h3><p class="c6"><span>The model expands on the concept of the previous model with two LSTM layers and dropout in between</span><span>. </span><span class="c4">&nbsp;</span></p><p class="c1"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 361.00px; height: 379.00px;"><img alt="" src="images/image4.png" style="width: 361.00px; height: 379.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c1"><span class="c4">Fig 3. GloVe + LSTM model architecture</span></p><p class="c1 c17"><span class="c4"></span></p><p class="c5"><span class="c4"></span></p><p class="c6"><span class="c4">Implemented in TensorFlow. </span></p><p class="c5"><span class="c4"></span></p><h3 class="c7" id="h.oiz8wxxkg38b"><span class="c14">GloVe + Bidirectional Long Short-term Memory (BiLSTM) </span></h3><p class="c6"><span class="c4">The model expands on the concept of the previous model with four LSTM layers and dropout in between. </span></p><p class="c5"><span class="c4"></span></p><p class="c1"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 361.00px; height: 506.00px;"><img alt="" src="images/image3.png" style="width: 361.00px; height: 506.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c1"><span class="c4">Fig 4. GloVe + BiLSTM model architecture</span></p><p class="c1 c17"><span class="c4"></span></p><p class="c6"><span class="c4">Implemented in TensorFlow. </span></p><p class="c5"><span class="c4"></span></p><h3 class="c7" id="h.dudj03teqmfe"><span class="c14">Deep Pyramid Convolutional Neural Networks (DPCNN)</span></h3><p class="c6"><span class="c4">Based on Johnson, Rie &amp; Zhang, Tong&rsquo;s DPCNN model[2] for text classification. </span></p><p class="c5"><span class="c4"></span></p><p class="c1"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 337.64px; height: 416.50px;"><img alt="" src="images/image5.png" style="width: 337.64px; height: 416.50px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c1"><span class="c4">Fig 5. DPCNN Architecture</span></p><p class="c1 c17"><span class="c4"></span></p><p class="c6"><span class="c4">Implemented in Tensorflow</span></p><p class="c5"><span class="c4"></span></p><h3 class="c7" id="h.ookqs83cx0o5"><span class="c14">Universal Language Model Fine-tuning for Text Classification (ULMFiT)</span></h3><p class="c6"><span class="c4">Based on Jeremy Howard and Sebastian Ruder&rsquo;s ULMFit model[3]. This model is state-of-the-art at the time of this writing. </span></p><p class="c5"><span class="c4"></span></p><p class="c1"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 602.00px; height: 237.33px;"><img alt="" src="images/image2.png" style="width: 602.00px; height: 237.33px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p><p class="c1"><span class="c4">Fig 6. ULMFiT model architecture</span></p><p class="c5"><span class="c4"></span></p><p class="c6"><span class="c4">Implemented in TensorFlow. </span></p><h2 class="c12" id="h.sqp27v926diz"><span class="c0">Training</span></h2><p class="c6"><span class="c4">The dataset was sliced into 98% training, 1% dev, and 1% testing. The model was trained on one P3.16xlarge with Deep Learning AMI. The model is saved after each epoch. </span></p><h2 class="c12" id="h.uf474jxf9qel"><span class="c0">Inference</span></h2><p class="c6"><span class="c4">The inference is done with a saved model in a Fargate container. </span></p><h2 class="c12" id="h.idosp1cl2s5l"><span class="c0">Deployment </span></h2><p class="c6"><span class="c4">Due to Lambda&rsquo;s limitation on size, I had to deploy the last model in a docker container on AWS Fargate.</span></p><p class="c5"><span class="c4"></span></p><h2 class="c12" id="h.4hw9jx6dz801"><span class="c0">Reference</span></h2><ol class="c3 lst-kix_6uaigt98ephh-0 start" start="1"><li class="c2"><span>Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. </span><span class="c8"><a class="c9" href="https://www.google.com/url?q=https://nlp.stanford.edu/pubs/glove.pdf&amp;sa=D&amp;ust=1552261754874000">GloVe: Global Vectors for Word Representation</a></span><span class="c4">.</span></li><li class="c2"><span>Johnson, Rie &amp; Zhang, Tong. 2017. </span><span class="c8"><a class="c9" href="https://www.google.com/url?q=https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf&amp;sa=D&amp;ust=1552261754874000">Deep Pyramid Convolutional Neural Networks for Text Categorization</a></span><span class="c4">.</span></li><li class="c2"><span>Jeremy Howard, Sebastian Ruder. 2018. </span><span class="c8"><a class="c9" href="https://www.google.com/url?q=https://arxiv.org/abs/1801.06146&amp;sa=D&amp;ust=1552261754875000">Universal Language Model Fine-tuning for Text Classification</a></span></li></ol><p class="c5"><span class="c4"></span></p><div><p class="c6"><span class="c15">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</span></p></div></html>
