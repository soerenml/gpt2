### 1 - Why should layer normalization be condonducted after the computational blocks?

The problem is what happens to your **original information** as it goes through each layer of the network.

***

#### The "Dirty" Way (Post-Layer Normalization)

Imagine you start with a piece of information, let's say a value of **10**. This value goes into a layer.

1.  The layer does some math and produces a new value, say **5**.
2.  The residual connection takes the **original value (10)** and the **new value (5)** and adds them together: `10 + 5 = 15`.
3.  **Here's the problem:** This result (**15**) is then normalized. That means it gets rescaled to a new, different value, maybe **2.5**.
4.  This new value (**2.5**) is passed to the next layer. The **original value of 10 is now gone**. It was combined and then transformed. Over many layers, the original information can get lost.

***

#### The "Clean" Way (Pre-Layer Normalization)

Now, let's start with the same **original value of 10**.

1.  Before the layer does its math, it **first normalizes the input**. The value **10** is converted to **1.2** for internal calculations.
2.  The layer then does its math using **1.2** and produces a new value, say **3**.
3.  The residual connection now takes the **original value (10)** and the **new value (3)** and adds them together: `10 + 3 = 13`.
4.  This new value (**13**) is passed directly to the next layer **without being changed**.

The key difference is that the **original information (10) is preserved** and never modified by the normalization. It travels straight through the network while the layers add their new information on top. This is what a **clean residual stream** means.

##### Summary

In case of a clean residual stream, we do not modify the original information until it is added to normalized outcome of the layer.

In the dirty case, the original information and the outcome of the layer are both normalized and added.

![Normalization](images/normalization.png)

### 2 - Why attention can be seen as a reduce- and MLP as a mapping function

Attention can be thought of as a **reduce** operation because it takes a sequence of inputs and aggregates them into a single, context-aware representation. The MLP, on the other hand, is a **mapping** function because it transforms that aggregated representation into a new, higher-dimensional space, acting on each element of the sequence independently.

***

#### Attention as a Reduce Operation

In a Transformer model, the attention mechanism takes a series of vectors (the tokens in a sentence) and calculates a weighted average of them. Each vector's contribution to the average is determined by its "attention score," which reflects its relevance to the other vectors in the sequence. The end result is a single, new vector for each token that summarizes information from the entire sequence.

For example, when the model is processing the word "bank," it uses attention to figure out if the word refers to a river bank or a financial institution. It does this by "looking at" and "reducing" the information from all the other words in the sentence to build a context-aware representation of "bank."

***

#### MLP as a Mapping Function

After the attention mechanism has created these context-rich representations, the MLP (Multi-Layer Perceptron) acts as a mapping function. It takes each of these new, aggregated vectors and transforms it. It processes each vector individually, performing a series of linear and non-linear operations (like a matrix multiplication followed by a ReLU activation function) to map it to a new vector space. This is where the model can learn and represent more complex, non-linear relationships in the data.

Unlike attention, which considers the entire sequence, the MLP operates on each token's vector independently, enriching its representation before it is passed on to the next layer of the network.