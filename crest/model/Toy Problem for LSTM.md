Toy Problem for LSTM
- A1 and A2 (LSTMs 1 and 2)
- A1 is taking in some feature X1 
- A2 is taking in some feature X2
- Base Case: Neither of them are taking a state feature from the other
- A1 is producing A1_0
- A2 is producing A2_0
- A2 taking in A1_0, X1, X2
- A1 taking in A2_0, X1, X2

A1_0 -> 1
A1_1 -> 4
A1_2 -> 4 + 3 + 4 => 11

A2_0 -> 1
A2_1 -> 4
A2_2 -> 11

