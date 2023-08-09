## A New Dataset of Handwritten Chinese Character Type for Classification

### Abstarct

In the Image and Video Recognition course during my master's studies, I undertook a project that focused on the creation and analysis of a novel dataset named CCT-4. This dataset comprises 400 distinct handwritten instances of Chinese characters, provided by five individual contributors. Historically, Chinese scripts have evolved considerably from their ancient forms to contemporary versions. For the construction of this dataset, I meticulously selected four distinctive scripts: Kin (bronze script), Ten (seal script), Kai (traditional Chinese), and Kan (simplified Chinese). Each script contains 20 unique characters, summing up to 80 characters across all four categories. Each participant contributed by writing all 80 characters spanning the four script types.

Subsequently, I employed a convolutional neural network to perform character classification on this dataset. Given the intricate nature of ancient Chinese scripts, the results were commendably accurate. This experiment aimed to establish a foundational benchmark for the efficacy of machine learning algorithms when applied to this dataset.


# ◉Dataset introduction
### Datset  category

The table shows Chinese, Chinese pronunciation, English and Japanese pronunciation of the four types of Chinese characters.   
The bronze script and the Seal script are ancient styles of writing Chinese characters.  
The bronze script is engraved in ritual bronzes such as big metal bell.  
In general, the Seal script originated from the Chinese bronze script.  
Traditional Chinese characters and Simplified Chinese characters are one type of standard Chinese character sets of the contemporary written Chinese.
<img src='./tmp/1-Category.png'>

### Image details
 400 samples = 20(character) × 4(category) × 5(people)

### List of Kan Chinese character
<img width= 400 src='./tmp/2-Character list.png'>
You can  click the following link to check the details of the characters

[Chinese_Character_List_Detail.pdf](Chinese_Character_List_Detail.pdf)

### Chinese character samples
<img width=400 src='./tmp/3-Printed Chinese characters.png'>

<img width=400 src='./tmp/4-Handwritten Chinese character.png'>




# ◉Classification Methods
I used 6 models to make a classification, as an attempt to provide a baseline for the performance of machine learning algorithms on the dataset.  
   (1) easyCNN  
   (2) ResNet18_Ft      
   (3) AlexNet  
   (4) GoogLeNet  
   (5) ResNet18  
   (6) RegNet_x_8gf  

### (1)easyCNN  
easyCNN is a basic model. In this model, the architecture consists of 5 layers : 2 convolutional layers and three fully-connected layers.

<img src='./tmp/5-EasyCNN.png'>
<br>
<br>

### **(2)Resnet18_Ft**
ResNet18_Ft is a modified model based on ResNet18.
I changed the kernels, stride and padding parameters of the first layer and reshaped the last layer.
<img width=400 src='./tmp/7-Resnet18.png'>


<img width=400 src='./tmp/6-Resnet18_Ft.png'>

### **(3)~(6)**  
From number 3 to number 6, these models are established state of art models. I reshaped the last layer the same as Resnet18_Ft.


# ◉Experiment
I performed experiments with Finetuning by the training details shown in the table, except easyCNN model.  

### **Training details**

<img width=350  src='./tmp/8-Training details.png'>

### **Implementation**
First, download the dataset from the following link.

[dataset](https://drive.google.com/file/d/1Kbn7hhfAlayL9FOlQNfVjvYYMbjb3EXU/view?usp=sharing)

Then, unzip the downloaded file and palce it same directory with the main.py

|-main.py  
|-datast

Finally, you can implement the main.py like the below.

    python main.py --model resnet18


* **option**:

--model (easycnn, alexnet, resnet18, googlenet, regnet_x_8gf, default=resnet18_ft)

--pretrained (False. default=True)

--optim (Adam, default=SGD)


### Experiment results
The empirical results underscored the superior efficacy of finetuning as opposed to training from scratch. Specifically, the ResNet18_Ft model outperformed its counterparts, manifesting a marked enhancement relative to the standard ResNet18.

While the training accuracy for all six models exceeded 94%, it is noteworthy that the test accuracy for easyCNN was a mere 22.5%. This discrepancy suggests a potential overfitting of the easyCNN model during its training phase. Meanwhile, GoogLeNet, AlexNet, RegNet_x_8gf, and ResNet18 yielded satisfactory test accuracy metrics.
<img src='./tmp/9-Results.png'>

# ◉Analysis
In our experimental outcomes, the highest test accuracy achieved across the six models was 72.5%. This result is less than optimal, and upon reflection, several underlying causes can be identified:

Firstly, the volume of training data available was limited. Chinese characters, inherently intricate in design, especially ancient scripts like the Bronze and Seal scripts, have numerous strokes, making their recognition a challenging task. Acquiring meaningful feature maps from a mere 400 images for such complex characters is a daunting undertaking for any model.

Additionally, the dataset contains characters that bear resemblance to one another, as illustrated in the subsequent figure. Enhancing the model's ability to discern nuanced differences between these similar characters is pivotal.

<img width=350 src='./tmp/10-Analysis example.png'>

Lastly, it should be noted that our cohort of five writers were not expert calligraphers. This lack of expertise was particularly evident when inscribing the Bronze and Seal scripts, with which they had minimal familiarity. Consequently, there were observable deviations between their handwritten samples and the established templates.
