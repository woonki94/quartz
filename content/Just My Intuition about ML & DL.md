

What is the purpose of ML?

Generally 
- There's a weight $w$ multiplied to features $x$. And there's a known labels $y$
- But usually these are in higher dimension rather than just 1 or 2-D space.

- So we design a problem to find a optimal w.
- For simple example, we design a optimization problem like		$$ \frac{1}{2}||Wx - y||^2$$ We call it loss function.

- Now we try to find W that minimizes this function.
- This is where we use gradient descent
	- First we need to get to know about the taylor approximation
	- we can approximate a nearby point using gradient and small alpha and direction
	- and if we generate a continuous points (sequence) using direction as grad, it is guaranteed to converge to at least local minimum ( Global convergence Theorem)



> When does the probability get in?