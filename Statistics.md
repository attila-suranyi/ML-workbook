# Statistics
### Reference
- Source:
	- 	An Introduction to Statistical Learning - Gareth James, Daniela Witten, Trevor Hastie & Robert Tibshirani
	- 	Valószínűségszámítás és matematikai statisztika - Obádovics J. Gyula

# Basic statistical concepts
## Sampling
- ### Types
	- Random: if all the samples have the same probability of getting chosen
	- Judgmental: várható szennyező források közelében érdemes mintázni pl
	- Systematic

- ### Confidence interval
	But how do we know if our sample is representative of the whole? Approximating with [[Probability#Normal Gauss distribution|normal distribution]], we can say that there is a certain probability (usually 95% or 99%) chance that the **true mean** of the population is within the confidence interval. To calculate the confidence interval, we need the mean and standard deviation of our sample, and a Z value from a table (?).^[https://www.mathsisfun.com/data/confidence-interval.html]
	
## Hypothesis testing^[https://bloomingtontutors.com/blog/when-to-use-the-z-test-versus-t-test]
- ### Student *t*-test^[https://www.youtube.com/watch?v=pTmLQvMM-1M]
	A *t-test* tells us if two means of some samples are **reliably** different. For example, we test a rice sample from one field, and another sample from another field. Is there significant difference between them, or the differences are just noise?
	We need calculate the *t-value*, which is a **signal to noise ratio**.
	![[t-value formula.png]]
	The gained t-test is then compared to a value, which tells us if we should reject our null hypotesis or no.
	
	Assumptions when you do a t-test:
			- samples has normal distribution
			- similar variance
			- same number of data points
	
	- #### T distribution
		https://www.investopedia.com/terms/t/tdistribution.asp

## Linear regression^[https://www.youtube.com/watch?v=zPG4NjIkCjc]
Our goal is to find the relation between an independent and a dependent value. If the correlation is linear, we can use linear regression. 
![[linear regression intro.png]]


# ISL
## 2. Statistical Learning
### 2.1 What is Statistical Learning?
- input variable / independent variable / X; output variable / dependent variable / Y

>Y = *f*(X) + *e*

Given an X input, there is an *f* function that provides us with systematic information about Y. The sign *e* is a random error term independent from X.

>In essence, statistical learning refers to a set of approaches for estimating *f*.

### 2.1.1 Why estimate *f*?
- #### Prediction 
	In case we are not necessarily interested in understanding the relationship between input and output, only in estimating the output. Like how exactly the 3 input variable relate to each other or the output, we don't care, only in getting the precise output.
	The accuracy of the prediction depends on **reducable and irreducable error**. Reducable error comes from choosing the correct method for our prediction, while irreducable error comes from for example unmeasured input variables.
	
- #### Inference: 
	In this case we want to understand the correlation between Y and X, and how Y cganhes as a function of X. Here *f* can not be a black box. 
	Questions to help understand this:	
	- Which inputs are associated with the response?
	- What is the relationship between the response and each predictors?
	- Is the relationship linear?
	
	Different methods can be appropriate for these goals. For example, a linear model offer a simple inference, but might won't be precise enough for predictions. In contrast, a more complex model might offer an accurate prediction, but it might be a less interpretable model.
	These differences between a rigid and flexible model are further discussed in [[#2 1 3 The trade-off between prediction accuracy and model interpretability|2.1.3]].
	
### 2.1.2 How do we predict *f*?

#### Parametric methods
1. Assumption about the functional form, or **choosing a model**, like linear: *f*(X) = ß<sub>0</sub> + ß<sub>1</sub>X<sub>1</sub> + ß<sub>2</sub>X<sub>2</sub> ....
2. After a model has been selected, we need to *fit* or *train* the model, which means estimating the parameters ß<sub>0</sub>, ß<sub>1</sub>...

	This approach reduces the task of estimating *f* to estimating a few parameters. The disadvantage is that the result highly depend on our model selection.
	We can address this problem by choosing a more *flexible* model. This requires more parameters, and can result the phenomenon called **overfitting**.
	
#### Overfitting:
This happens when the model follows errors, or noise, too closely. In terms ML (machine learning) and neural networks, a model is overfitting when it is good at classifying **trainig data**, but not so good at **test data**. It means that the model is not good at generalizing data. It learned the features of the training data too well, worked too hard on finding patterns in it, which might not even be there, just caused by random chance. And if we give it data that is slightly different than the training data, it can't classify it properly.^[https://www.youtube.com/watch?v=DEMmkFC6IGM]

#### Non-parametric methods
Instead of starting from choosing a model, or making an assumption about the explicit form of *f*, like the linear form we've seen above, non-parametric methods seek an estimate of *f* that gets as close to the data points as possible, without being too rough or wiggly.
	The *smoothness* of the estimation has to be determined, meaning how closely it follows the data. Choosing a lower level of smoothness can result in of overfitting.
	
**Advantage**: this way the resulting estimate can fit a wide range of possible shapes of *f*.
**Disadvantage**: since this approach does not reduce the problem to estimating parameters, a large number of observations are needed.

Why the need for so many models? No one method dominates all others over all possible data sets.
	
### 2.1.3 The trade-off between prediction accuracy and model interpretability
When we decide between a more flexible, or a restrictive model, we have to ask ourselves what is our goal, **prediction or inference**?
A more restrictive, simpler model can help us interpret and understand the data and its relations better, but will be less capable of prediction.
On the other hand, flexible approaches can lead to such complicated estimates that it is difficult to understand the relations.

When we care about predictions, flexible methods are usually better, excpet if we [[Computer Science/Machine_Learning/Statistics#Overfitting|overfit]].
	
## 2.2 Model accuracy
- ### Regression models
	Regression is about **quantitative** information.
	
	- Measuring fit: *mean squared error* (MSE), which measures how close our prediction is to a true response value:
		![[mean squared error.png]]
	- When we test the accuracy of our model, we don't test on data set which was used for training our model. 
		>There is no guarantee that the method with the lowest training MSE will also have the lowest MSE.
	
		This can be understood if we think about [[Computer Science/Machine_Learning/Statistics#Overfitting|overfitting]]. 
	
	- #### Accuracy depending on model flexibility
		![[ISL accuracy vs model flexibility.png]]
		On the left panel different models were fitted. Since this is a simulated case, we know the true form of *f*, which is the **green** curve.
		On the right panel we see how the accuracy depends on the model flexibility. The grey curve shows the training MSE, and the more flexible the model, the smaller it will be. 
		**However**, looking at the **red** curve shows the test MSE, it can be seen that neither the most rigid nor the most flexible model reach the lowest test MSE.

### 2.2.2 The Bias- Variance Trade-Off

There is a middle case, sitting at the bottom of the U shaped curve. 
This **U shape** is the result of two competing properties, or errors, which add to the MSE:
		
- #### Variance
	>Variance refers to the amount by which *f* would change if we estimated it using a different training data set.
			
A method which follows the training data more closely, will change more on changing the data -> flexible methods have higher variance.
- #### Bias
	>Bias refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model.
				
![[Pasted image 20220407174350.png|400]]


A simpler, more rigid method has higher bias.
			
As mentioned, these errors are opposing each other, so minimizing one of them will result the increase of the other. This is called the **bias-variance trade-off**. The optimal approach lies somewhere middle ground, at the bottom of the U shape.
			
There are problems / data sets, which are more vulnerable to one or the other error. For example, in a case of a complex problem, a rigid method will have a higher bias.