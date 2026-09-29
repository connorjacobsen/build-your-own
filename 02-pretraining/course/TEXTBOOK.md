# What pretraining optimizes

An autoregressive model defines p(x_1,...,x_T)=product_t p(x_t | x_<t). Training minimizes the negative log likelihood of observed targets. Teacher forcing supplies the real preceding tokens; generation instead feeds back sampled tokens. This difference helps explain why a low training loss does not guarantee robust long continuations.

## One shifted sequence

For token IDs [BOS,a,b,EOS], the inputs [BOS,a,b] predict [a,b,EOS]. The logit at b predicts EOS. Appending EOS to a generation prompt would tell the model that the document has already ended; the training and inference boundaries therefore differ deliberately.

With batches of unequal sequences, the token objective is the sum of all target losses divided by the total target count. If one sequence contributes one token with loss 4 and another contributes three tokens with total loss 3, the correct mean is 7/4=1.75. Averaging sequence means gives (4+1)/2=2.5, a different objective.

## Decoder equations

For each layer, normalize the residual stream, project Q/K/V, rotate Q/K by absolute position, compute softmax(QK^T/sqrt(d)+causal_mask)V, project back and add the residual. Then normalize again and add W_down(SiLU(W_gate z) * W_up z). The final normalization and output projection produce raw vocabulary logits.

GQA stores fewer KV heads than query heads. It changes the size of the state reused by inference without requiring every query head to share the same attention distribution. The checkpoint contract fixes head mapping and rotary pairing, so shapes alone are insufficient for compatibility.

## Optimization and state

Adam's moments obey m_t=beta1*m_(t-1)+(1-beta1)*g_t and v_t=beta2*v_(t-1)+(1-beta2)*g_t^2. Bias correction divides by 1-beta^t. AdamW separately scales parameters by 1-lr*weight_decay before the adaptive update. A checkpoint missing moments or the update count cannot faithfully resume that optimizer.

First diagnose whether a tiny fixed batch can be learned. Then separate train, validation and final test data. Use validation for choices such as learning rate and stopping point. Repeatedly selecting on test results turns the test set into another validation set.

This model is intentionally tiny and byte-based. It teaches the mechanics of pretraining rather than reproducing the capabilities or scale of a production foundation model. Exporting its exact architecture to the inference course is a stronger systems exercise than displaying a plausible sentence.
