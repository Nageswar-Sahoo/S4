
# S4

Part 1 :

  Neural Network Architecture 
 
  ![image](https://user-images.githubusercontent.com/53977148/137504158-274818a2-750e-424e-b49b-1e9ca7273972.png)


  Forward Propagation Details 
  
    We can calculate an output from a neural network by propagating an input signal through each layer until the output layer outputs its values.We call this forward-       propagation.We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer.

    input1=i1
    input2=i2
    actual_output=t1
    actual_output=t2
    h1=i1*w1+i2*w2
    h2=i1*w3+i2*w4
    act_h1=sigmoid(h1)
    act_h2=sigmoid(h2)
    o1= act_h1*w5 + act_h2*w6
    o2= act_h1*w7 + act_h2*w8
    act_o1=sigmoid(o1)
    act_o2=sigmoid(o2)
    E1=1/2*( t1 - act_o1)2
    E2=1/2*(t2 - act_o2)2
    E_total=E1+E2
    sigmoid(x)=(1/1+exp(-x))


  Back Propagation Details
  
  
The backpropagation algorithm is named for the way in which weights are trained.Error is calculated between the actual outputs labels and the outputs labels forward propagated from the network. These errors are then propagated backward through the network from the output layer to the hidden layer, updating weights as they go.


    dsigmoid(x)/dx=sigmoid(x)*(1-sigmoid(x))
    dEtotal/dw5= d(E1+E2)/dw5= dE1/dact_o1*dact_o1/do1*do1/dw5	
    dE1/dact_o1=0.5*2*(t1 - act_o1)*-1 =act_o1 - t1		
    dact_o1/do1=dsigmoid(o1)/do1 =  act_o1 * (1-act_o1)		
    do1/dw5=d(act_h1*dw5)/dw5+d(act_h2*dw6)/dw5=act_h1*1+act_h2*0=act_h1	
    dE1/dact_o1=0.5*2*(t1 - act_o1)*-1 =act_o1 - t1
    dact_o1/do1=dsigmoid(o1)/do1 =  act_o1 * (1-act_o1)	
    do1/dw6=d(act_h1*dw5)/dw6+d(act_h2*dw6)/dw6=act_h1*0+act_h2*1=act_h2 
    dEtotal/dw7= d(E1+E2)/dw7= dE2/dact_o2*dact_o2/do2*do1/dw7						
    dE2/dact_o2=0.5*2*(t2 - act_o2)*-1 =act_o2 - t2						
    dact_o2/do2=dsigmoid(o2)/do2 =  act_o2 * (1-act_o2)						
    do2/dw7=d(act_h1*dw7)/dw7+d(act_h2*dw6)/dw7=act_h1*1+act_h2*0=act_h1						
					
    dEtotal/dw8= d(E1+E2)/dw8= dE2/dact_o2*dact_o2/do2*do1/dw8					
    dE2/dact_o2=0.5*2*(t2 - act_o2)*-1 =act_o2 - t2					
    dact_o2/do2=dsigmoid(o2)/do2 =  act_o2 * (1-act_o2)					
    do2/dw8=d(act_h1*dw7)/dw8+d(act_h2*dw6)/dw8=act_h1*0+act_h2*1=act_h2					

    do1/dact_h1=d(act_h1*w5+act_h2*w6)/dact_h1= w5							
    do1/dact_h2=d(act_h1*w5+act_h2*w6)/dact_h2= w6							
    dE1/dact_h1=   dE1/dact_o1*dact_o1/do1*do1/dact_h1  =  (act_o1 - t1)*act_o1*(1-act_o1) *w5							
    dE1/dact_h2=   dE1/dact_o1*dact_o1/do1*do1/dact_h2  =  (act_o1 - t1)*act_o1*(1-act_o1) *w6		
    do2/dact_h1=d(act_h1*w7+act_h2*w8)/dact_h1= w7							
    do2/dact_h2=d(act_h1*w7+act_h2*w8)/dact_h2= w8							
    dE2/dact_h1=   dE2/dact_o2*dact_o2/do2*do2/dact_h1  =  (act_o2 - t2)*act_o2*(1-act_o2) *w7							
    dE2/dact_h2=   dE2/dact_o2*dact_o2/do2*do1/dact_h2  =  (act_o2 - t2)*act_o2*(1-act_o2) *w8					
    dEtotal/dact_h1= d(E1+E2)/dact_h1= dE1/dact_h1+ dE2/dact_h1=     (act_o1 - t1)*act_o1*(1-act_o1) *w5 + (act_o2 - t2)*act_o2*(1-act_o2) *w7										
    dEtotal/dact_h2= d(E1+E2)/dact_h2= dE1/dact_h2+ dE2/dact_h2=     (act_o1 - t1)*act_o1*(1-act_o1) *w6 + (act_o2 - t2)*act_o2*(1-act_o2) *w8										
	dact_h1/dh1= act_h1*(1-act_h1)
    dact_h2/dh2= act_h2*(1-act_h2)   
    dh1/dw1=i1	
    dh1/dw2=i2		
    dh2/dw3=i1		
    dh2/dw4=i2	

    dEtotal/dw1 = ((act_o1 - t1)*act_o1*(1-act_o1) *w5 + (act_o2 - t2)*act_o2*(1-act_o2) *w7) * act_h1*(1-act_h1)*i1						
    dEtotal/dw2 = ((act_o1 - t1)*act_o1*(1-act_o1) *w5 + (act_o2 - t2)*act_o2*(1-act_o2) *w7) * act_h1*(1-act_h1)*i2						
    dEtotal/dw3 = (act_o1 - t1)*act_o1*(1-act_o1) *w6 + (act_o2 - t2)*act_o2*(1-act_o2) *w8) *act_h2*(1-act_h2) * i1						
    dEtotal/dw4 = (act_o1 - t1)*act_o1*(1-act_o1) *w6 + (act_o2 - t2)*act_o2*(1-act_o2) *w8) *act_h2*(1-act_h2) * i2						
    dEtotal/dw5 = (act_o1 - t1)*act_o1*(1-act_o1)*act_h1 	
    dEtotal/dw6 = (act_o1 - t1)*act_o1*(1-act_o1)*act_h2 
    dEtotal/dw7 = (act_o2 - t2)*act_o2*(1-act_o2)*act_h1 						
	dEtotal/dw8 = (act_o2 - t2)*act_o2*(1-act_o2)*act_h2 					

Image from Excel 

![Capture](https://user-images.githubusercontent.com/53977148/137503326-f124687a-bdb6-4c0f-bd37-c37c1c743d61.PNG)

![Capture1](https://user-images.githubusercontent.com/53977148/137502273-740bb820-a4b2-41fb-a3d1-638d3b3102e6.PNG)

![Capture3](https://user-images.githubusercontent.com/53977148/137502016-82f599d9-7fb5-4ebf-8774-329228560a62.PNG)

![Capture4](https://user-images.githubusercontent.com/53977148/137502046-7afa8b28-4b3e-4db7-83d2-b8b8b702d35c.PNG)

![Capture5](https://user-images.githubusercontent.com/53977148/137502060-660f36bf-c614-4ac2-8093-bafa43dd6135.PNG)

![Capture6](https://user-images.githubusercontent.com/53977148/137502115-f542bae8-6322-42b6-828e-c47006a88cc9.PNG)

![Capture7](https://user-images.githubusercontent.com/53977148/137502163-a6e5d92d-ecea-4e3e-8e33-3638a46a1417.PNG)

Error graph with Learning Rate 0.1 

![lr_0 1](https://user-images.githubusercontent.com/53977148/137502861-f3c1483a-316e-487a-a65f-6b891c50348c.PNG)

Error graph with Learning Rate 0.2

![lr_0 2](https://user-images.githubusercontent.com/53977148/137502866-34cc2084-9784-4020-bec9-cb1139c081df.PNG)

Error graph with Learning Rate 0.5 

![lr_0 5](https://user-images.githubusercontent.com/53977148/137502873-1dc2ac1b-d102-4e3e-92a6-c8e841579d94.PNG)

Error graph with Learning Rate 0.8

![lr_0 8](https://user-images.githubusercontent.com/53977148/137503028-7148e8ff-d5ad-4fb5-ac37-5fb8a002f30d.PNG)

Error graph with Learning Rate 1 

![lr_1](https://user-images.githubusercontent.com/53977148/137503040-f8bd63da-c4ef-462e-a30e-e93db96a68fe.PNG)

Error graph with Learning Rate 2 

![lr_2](https://user-images.githubusercontent.com/53977148/137503045-0d059ed6-a137-4c3a-a692-958ad6edd180.PNG)
