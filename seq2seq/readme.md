#  Sequence to Sequence Learning
模型总结：
+ 使用seq2seq最基本的形式
+ 没有softmax
+ 没有attention
+ 没有对多余的pad处理，如pad_pack
+ 没有设置stop
+ 莫名其妙的对输入数据dropout
+ 没有使用teacher_forcing预热
+ 没有使用标准的损失函数如rouge,belu之类的，简单的忽视pad位，进行交叉熵损失
+ 若是考虑了rouge_belu这种则要考虑整体的损失函数问题，涉及到greedy_search和beam search
+ 

