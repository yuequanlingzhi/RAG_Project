# 基本介绍

本项目采用rag(**retrieval-augmented generation**)实现

主要流程为：

1.  加载和处理PDF文件
2. 文本嵌入与向量检索（主要就是将文档分成及各个chunk，每个chunk编码为一个向量，通过计算问题与每个向量之间的L2距离检索和问题相关的文段）
3. 将得到的文段作为prompt丢入大模型，让大模型输出问题的答案

 # 注意事项
记得将Final_Project_Documents文件夹单独导入，路径随便，只要更改rag.ipynb中下面的代码就行了
```
pdf_dir = 'Final_Project_Documents/test' #YOUR_PDF_DIR_PATH
```
可以随便挑个小的pdf文件作为测试，放到新建的test文件夹中

代码中采用了简易的gpt2作为大模型，效果不咋样，可能还需要调整，或者到时候直接把prompt丢入现成的大模型，又或者人肉搜索检索出来的文段得到答案。

可能需要安装的依赖
***

```
pip install transformers faiss-cpu sentence-transformers PyPDF2 
```
***
我试了下用SimCES来微调，效果很容易过拟合，而且需要较大的显存，时间关系没做多少次测试，可以考虑更换成更简易的预训练模型，然后微调的时候epoch和Lr调小一点看一下效果怎么样。这个项目主要还是需要**文本嵌入与向量检索**这一步的准确率
***
代码运行需要从huggingface下载预训练权重，一般需要科学上网
