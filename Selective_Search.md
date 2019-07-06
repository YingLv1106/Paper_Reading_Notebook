
# Selective Search for Object Recognition

title: **Selective Search**  
date: 2019-07-05 10:18  
tags: **CNNs for Ovject Recognition**
###Introduction
1) **Selective Search:**
### **Architecture**

**Fig.1 Object detection system overview**

- Freams  localization as a regression problem.
- **Goal:** to generate a class-independent, data-driven, selective search strategy that generates a small set of high-quality object locations.  
### **Contributions**
1) 將分割作爲一種選擇性搜索策略進行調整, 有哪些好的多樣化策略？
2) 選擇性搜索對於在圖像中創建一小組高質量的位置有多有效？
3) 我們是否可以使用選擇性搜索來使用更強大的分類器和外觀模型來進行對象識別？  
##**3. Select search**
將兩個最相似的區域組合在一起, 幷計算得到的區域與其相鄰區域之間的新相似性.  
####**Follow shown Hierarchical Grouping Algorithm**
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703160008569-1566027026.png">
</div>

###Diversification Strategies(抽樣多樣化)

- 1）使用各種具有不同不變性的顏色空間。
- 2）使用不同相似度。
- 3）改變起始區域。  

####**A.** Complementary Colour Spaces

---
- 顏色不变性定义：室外光线的彩色成分变化非常大，但人却能正确的感知
场景中物体的颜色，并且在大部分情况下不依赖于环境照明的颜色，这种现象叫彩色不变性。  
####**B.** 四種相似性度量方法:
我们在计算多种相似度的时候，都是把单一相似度的值归一化到[0,1]之间，1表示两个区域之间相似度最大。  
- 颜色相似度  
使用L1-norm归一化获取图像每个颜色通道的25bins的直方图，这样每个区域都可以
得到一个75维的向量，区域之间颜色相似度通过下面的公式计算：
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703160821813-1076817497.png"> 
</div> 
上面这个公式可能你第一眼看过去看不懂，那咱们打个比方，由于
Ci,Cj是归一化后值，
每一个颜色通道的直方图累加和为1.0，三个通道的累加和就为3.0，
如果区域ci和区域cj直方图完全一样，则此时颜色相似度最大为3.0，如果不一样，
由于累加取两个区域bin的最小值进行累加，当直方图差距越大，
累加的和就会越小，即颜色相似度越小。

在区域合并过程中使用需要对新的区域进行计算其直方图，计算方法：
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703160938444-973518030.png">
</div>  

- 紋理相似度
这里的纹理采用SIFT-Like特征。具体做法是对每个颜色通道的8个不同方向计算方差σ=1的高斯微分（Gaussian Derivative），
使用L1-norm归一化获取图像每个颜色通道的每个方向的10 bins的直方图，这样就可以获取到一个240（10x8x3）维的向量
，区域之间纹理相似度计算方式和颜色相似度计算方式类似，合并之后新区域的纹理特征计算方式和颜色特征计算相同：
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703161140898-785370498.png">
</div>   

- 优先合并小的区域  
如果仅仅是通过颜色和纹理特征合并的话，很容易使得合并后的区域不断吞并周围的区域，
后果就是多尺度只应用在了那个局部，而不是全局的多尺度。因此我们给小的区域更多的
权重，这样保证在图像每个位置都是多尺度的在合并。
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703161426901-1588964563.png">
</div>
上面的公式表示，两个区域越小，其相似度越大，越接近1。  

- 区域的合适度距离
如果区域ri包含在rj内，我们首先应该合并，另一方面，如果ri很难与rj相接，他们之间
会形成断崖，不应该合并在一块。这里定义区域的合适度距离主要是为了衡量两个区域是
否更加“吻合”，其指标是合并后的区域的Bounding Box（能够框住区域的最小矩形
BBij）越小，其吻合度越高，即相似度越接近1。其计算方式：
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703162700125-1235760831.png">
</div>

- 合并上面四种相似度
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703163551711-370451075.png">
</div>

####**C.** 給區域打分
通过上述的步骤我们能够得到很多很多的区域，但是显然不是每个区域作为目标的
可能性都是相同的，因此我们需要衡量这个可能性，这样就可以根据我们的需要筛
选区域建议个数啦。  
这篇文章做法是，给予最先合并的图片块较大的权重，比如最后一块完整图像权重
为1，倒数第二次合并的区域权重为2以此类推。但是当我们策略很多，多样性很多
的时候呢，这个权重就会有太多的重合了，排序不好搞啊。文章做法是给他们乘以
一个随机数，毕竟3分看运气嘛，然后对于相同的区域多次出现的也叠加下权重，毕
竟多个方法都说你是目标，也是有理由的嘛。这样我就得到了所有区域的目标分数，
也就可以根据自己的需要选择需要多少个区域了。

####**D.** 选择性搜索性能评估

自然地，通过算法计算得到的包含物体的Bounding Boxes与真实情况（ground truth）
的窗口重叠越多，那么算法性能就越好。这是使用的指标是平均最高重叠率ABO
（Average Best Overlap）。对于每个固定的类别 c，每个真实情况
（ground truth）表示为 ，令计算得到的位置假设L中的每个值lj，那么 
ABO的公式表达为：
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703164314166-380531713.png">
</div>
重叠率的计算方式：  
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703164333093-1026209796.png">
</div>
上面结果给出的是一个类别的ABO，对于所有类别下的性能评价，很自然就是使用所有
类别的ABO的平均值MABO（Mean Average Best Overlap）来评价。  

#####**1.** 單一策略評估
我们可以通过改变多样性策略中的任何一种，评估选择性搜索的MABO性能指标。论文
中采取的策略如下：  
- 使用RGB色彩空间(基于图的图像分割会利用不同的色彩进行图像区域分割)
- 采用四种相似度计算的组合方式
- 设置图像分割的阈值k=50
然后通过改变其中一个策略参数，获取MABO性能指标如下表(第一列为改变的参数，
第二列为MABO值，第三列为获取的候选区的个数)：
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703170529405-1261512224.png">
</div>

表中左侧为不同的相似度组合，单独的，我们可以看到纹理相似度表现最差，MABO
为0.581，其他的MABO值介于0.63和0.64之间。当使用多种相似度组合时MABO性
能优于单种相似度。表的右上角表名使用HSV颜色空间，有463个候选区域，而且MABO
值最大为0.693。表的右下角表名使用较小的阈值，会得到更多的候选区和较高
的MABO值。  

#####**2.** 多樣化策略組合
我们使用贪婪的搜索算法，把单一策略进行组合，会获得较高的MABO，但是也会造成
计算成本的增加。下表给出了三种组合的MABO性能指标：
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703183204190-1407191313.png">
</div>
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703185945645-1480942386.png">
</div>

上图中的绿色边框为对象的标记边框，红色边框为我们使用 'Quality' Selective 
Search算法获得的Overlap最高的候选框。可以看到我们这个候选框和真实标记非常
接近。  
下表为和其它算法在VOC 2007测试集上的比较结果：
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703190805553-853063658.png">
</div>  

下图为各个算法在选取不同候选区数量，Recall和MABO性能的曲线图，从计算成本、
以及性能考虑，Selective Search Fast算法在2000个候选区时，效果较好。
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703190842002-332653310.png">
</div>  

####**D.** 代碼實現
 我们可以通过下面命令直接安装Selective Search包。
- `pip install selectivesearch`   
然后从[follow](https://github.com/AlpacaDB/selectivesearch)下载源码，运行example\example.py文件。效果如下：
```
# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np


def main():

    # 加载图片数据
    img = skimage.data.astronaut() 

    '''
    执行selective search，regions格式如下
    [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
    ]
    '''
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    #计算一共分割了多少个原始候选区域
    temp = set()
    for i in range(img_lbl.shape[0]):
        for j in range(img_lbl.shape[1]):    
            temp.add(img_lbl[i,j,3]) 
    print(len(temp))       #286
    
    #计算利用Selective Search算法得到了多少个候选区域
    print(len(regions))    #570
    #创建一个集合 元素不会重复，每一个元素都是一个list(左上角x，左上角y,宽,高)，表示一个候选区域的边框
    candidates = set()
    for r in regions:
        #排除重复的候选区
        if r['rect'] in candidates:
            continue
        #排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)  
        if r['size'] < 2000:
            continue
        #排除扭曲的候选区域边框  即只保留近似正方形的
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    #在原始图像上绘制候选区域边框
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()
    

if __name__ == "__main__":
    main()
```  
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703231142451-315466307.png">
</div>  
<div align="center">
<img src="https://images2018.cnblogs.com/blog/1328274/201807/1328274-20180703231125031-1308002478.png">
</div>  

selective_search函数的定义如下：
```
    def selective_search(
        im_orig, scale=1.0, sigma=0.8, min_size=50):
    '''Selective Search

    首先通过基于图的图像分割方法初始化原始区域，就是将图像分割成很多很多的小块
    然后我们使用贪心策略，计算每两个相邻的区域的相似度
    然后每次合并最相似的两块，直到最终只剩下一块完整的图片
    然后这其中每次产生的图像块包括合并的图像块我们都保存下来
    
    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size 候选区域大小，并不是边框的大小
                },
                ...
            ]
    '''
    assert im_orig.shape[2] == 3, "3ch image is expected"

    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)] 
    #图片分割 把候选区域标签合并到最后一个通道上 height x width x 4  每一个像素的值为[r,g,b,(region)] 
    img = _generate_segments(im_orig, scale, sigma, min_size)

    if img is None:
        return None, {}

    #计算图像大小
    imsize = img.shape[0] * img.shape[1]
    
    #dict类型，键值为候选区域的标签   值为候选区域的信息，包括候选区域的边框，以及区域的大小，颜色直方图，纹理特征直方图等信息
    R = _extract_regions(img)

    #list类型 每一个元素都是邻居候选区域对(ri,rj)  (即两两相交的候选区域)
    neighbours = _extract_neighbours(R)

    # calculate initial similarities 初始化相似集合S = ϕ
    S = {}
    #计算每一个邻居候选区域对的相似度s(ri,rj)
    for (ai, ar), (bi, br) in neighbours:      
        #S=S∪s(ri,rj)  ai表示候选区域ar的标签  比如当ai=1 bi=2 S[(1,2)就表示候选区域1和候选区域2的相似度
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # hierarchal search 层次搜索 直至相似度集合为空
    while S != {}:

        # get highest similarity  获取相似度最高的两个候选区域  i,j表示候选区域标签
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]   #按照相似度排序

        # merge corresponding regions  合并相似度最高的两个邻居候选区域 rt = ri∪rj ,R = R∪rt
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # mark similarities for regions to be removed   获取需要删除的元素的键值        
        key_to_delete = []         
        for k, v in S.items():       #k表示邻居候选区域对(i,j)  v表示候选区域(i,j)表示相似度
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # remove old similarities of related regions 移除候选区域ri对应的所有相似度：S = S\s(ri,r*)  移除候选区域rj对应的所有相似度：S = S\s(r*,rj)
        for k in key_to_delete:
            del S[k]

        # calculate similarity set with the new region  计算候选区域rt对应的相似度集合St,S = S∪St
        for k in filter(lambda a: a != (i, j), key_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    #获取每一个候选区域的的信息  边框、以及候选区域size,标签
    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })
            
    #img：基于图的图像分割得到的候选区域   regions：Selective Search算法得到的候选区域
    return img, regions
```