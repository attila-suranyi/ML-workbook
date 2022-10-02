# Neural Networks from Scratch

- **Motivation**: Deep understanding of neural networks. This enables to switch easier between frameworks, like Keras or PyTorch. 
- ![[Pasted image 20220916195703.png|1300]]
- Biases: sum(neurons)
- Weights: connections between each layer, (10 * 16) +  (16 * 16) + ...
-----
- ![[Pasted image 20220916203345.png]]
- *output = (input * weights) + bias*
	![[Pasted image 20220916203821.png]]
- You cant really change the **inputs**, these can be the actual input data from sensors, or calculated values from another layer. You change **weights and biases**. 
	``` python
	some_value = -0.5
	weight = 0.7
	bias = 0.7
	
	print(some_value * weight) # =-0.35 -> changing the magnitude
	print(some_value + bias)   # = 0.2  -> off sets the value 
	```
- Think of the equation of a line: *y = ax + m*. Weight gonna change 
	![[Pasted image 20220916223224.png]]
- https://preview.redd.it/10msjwkgp9o91.gif?format=mp4&s=d405cbe9d0c63762a0e5c1683161fe8c5986546f
- ![[Pasted image 20220916215630.png]]
 
- [[Linear algebra#Dot product|Dot product]]: 
	- **Neuron** (two vectors): `output = (input * weights) + bias` -> dot(input, weights) + bias
	- **Layer**: more weights, so its a matrix now. Dimension translates to number of neurons in the layer. [[Linear algebra#Linear systems of equations|Linear systems of equations]]. We calculate dot product as many times as many neurons there are. The result is a **vector**!
	- **Batch**