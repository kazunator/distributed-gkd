# distributed-gkd
There is still an error in the seq length when you go from the first forward pass to the second. The issue could be on the dataloader, but it manifests itself in trainer.py at the function teacher_forward_second_half. Use prints to locate it again, but printing the shapes of the tensors in gpu0 and gpu1.
