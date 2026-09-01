
# AI-Use Appendix - Ritika Mukesh Neema

## 1. Which parts did you use an assistant for, and which did you write yourself?

I used Claude to help scaffold the PyTorch and TensorFlow training code,
the CUDA matmul.cu source file, and the data visualization cells. I wrote
the parameter calculations, data loading, markdown analysis/conclusions,
and the cross-framework comparison myself. I also debugged the PyTorch
training issue (described below) by working through the diagnosis
interactively with the assistant.

## 2. Give one specific thing it produced that was wrong.

The initial PyTorch training function used full-batch gradient descent
(passing all 531 training samples as one batch per epoch), while the
TensorFlow version used mini-batches of 32. This caused the PyTorch
baseline to get stuck predicting only the majority class. The output was:

    PyTorch Baseline: 0.6228 ± 0.0000
    PyTorch Modified: 0.7135 ± 0.0338

The baseline accuracy of 0.6228 exactly matched the proportion of class 1
in the test set (y_test.mean() = 0.6228), and the standard deviation of
0.0000 across three seeds confirmed the model was not learning — it was
outputting the same prediction every time regardless of input.

## 3. How did you find out? What did the failure look like?

The std of 0.0000 was the first red flag — three different random seeds
should not produce identical accuracy. I added a diagnostic cell to check
the raw predictions of an untrained model:

    Prediction mean: 0.4752, min: 0.4655, max: 0.4839
    Test set class balance: 0.6228

All predictions clustered around 0.475, meaning after training the model
simply learned to push all outputs above 0.5 (predict all 1s) to minimize
loss, achieving exactly the majority-class rate. The model was not
distinguishing between classes at all.

## 4. What did you change, and why does your version work?

I replaced the full-batch forward pass with a PyTorch DataLoader using
batch_size=32 and shuffle=True, matching the TensorFlow configuration:

    train_ds = TensorDataset(Xt, yt)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

Mini-batch SGD introduces noise in the gradient estimates, which helps the
optimizer escape flat regions of the loss landscape like the majority-class
plateau. The shuffled mini-batches also mean each update sees a different
class distribution, preventing the model from settling on a trivial
solution. After the fix:

    PyTorch Baseline: 0.7456 ± 0.0189
    PyTorch Modified: 0.7485 ± 0.0041

Both models now show meaningful learning with non-zero variance across seeds.