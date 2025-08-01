

unlike building a vocab for rnn's

for gpt2 like transformer architecture, we only need a tokenized data with eos

but we cannot load all data at once, first store the tokenized data and  using memmap do lazy fetch of saved file, to load onto dataloader.

then train for that partial data-> fetch again and so on



