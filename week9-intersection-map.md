#### 1. What does pseudo-labeling / confidence-filtering become when the model is zero-shot CLIP?

Answer:

conf [0.0,0.1): n= 620 acc=0.106  
conf [0.1,0.2): n=2072 acc=0.208  
conf [0.2,0.3): n=2283 acc=0.328  
conf [0.3,0.4): n=2225 acc=0.464  
conf [0.4,0.5): n=2268 acc=0.545  
conf [0.5,0.6): n=2229 acc=0.651  
conf [0.6,0.7): n=2165 acc=0.763  
conf [0.7,0.8): n=2385 acc=0.851  
conf [0.8,0.9): n=3321 acc=0.922  
conf [0.9,1.0): n=10432 acc=0.987

High-confidence pseudo-labeling on zero-shot CLIP is safe — arguably overly conservative. At the 90%+ threshold, you'd get 98.7% clean labels. But because CLIP underestimates its own accuracy, a naive high-confidence filter throws away a lot of correct predictions sitting in the 0.6–0.9 confidence range (76–92% accurate) that a properly calibrated filter would keep. So the honest answer isn't "pseudo-labeling works" or "doesn't work" — it's "works, but a fixed confidence threshold is the wrong filter for an underconfident model; you're leaving good labels on the table."
