Project self-organised fenoes HMM model 
Progress Report on 19.2.2020

Work has been done: 
1. we have already find all the mfcc-vectors for each frame for the dataset tedlium s5_r2. But theses mfcc-vectors are all 13-diemensional vectors, so I have applied a cmvn and a add-deltas functions to get 39-dimensional mfcc-vectors with 0 mean und variance 1.
2. Then I have found a alignment file and done a ali-to-phone function do get the phones for each frame.
3. Afterward, I wrote a extract_data.py to categorize every mfcc-vectors sequece for each phone.
4. Now we have for each phone a mfcc-vectors sequences e.g. phone /SIL/ [sequence1, sequence2, .......sequence1000] and every sequence in the list may have different size and size is depending on the frame-length for each sequence]
Problem we have now:
1. For each phone, every sequence-length is very different, some have 100 frames but some just have 3 frames. How to make the length equally?
2. After length-equalization, we will put every frame vector into a DNN to get a probability and multiply them together, But some phones like SIL have a lot of sequence e.g 11000+, so after multiplcation, the number will be extremely small.
3. Some phone like SIL have a lot of sequence but some phone don't show very frequent and some phone don't even show in the dataset. How to solve this question?

