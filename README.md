# Spatial Transformer for 3D Point Clouds

## Introdution 

This repository is a rewrite spatial transformer from [Spatial Transformer for 3D Point Clouds](https://github.com/samaonline/spatial-transformer-for-3d-point-clouds),

because that repo build by  Tensorflow, and frame mixed another task and spatial transformer,

i decided refer to that repo write pure spatial transformer - [Spatial_transformer.py].

Contribution of this repo is rewrite to pytoech frame and package to class,
very friendly to use.



因為參考的 repo 把 Spatial Transformer 跟下游任務寫在一起，

而且是用 Tesorflow 建的，我這邊不方便使用。

所以決定將含有 Spatial Transformer 的部分拔出來重寫成 pytorch 版本，
並且包成 class 出來用。



**簡單來說這包是 [Spatial Transformer for 3D Point Clouds](https://github.com/samaonline/spatial-transformer-for-3d-point-clouds) 的 Spatial Transformer 部分移植 pytorch 版。**


## Quick Start


Step 1

In your model file import [Spatial_transformer.py]

```python
from Spatial_transformer import offset_deform as OD
```

Step 2 

In your model class create a object **(optional)**.

```python
self.OD = OD()
```

Step 3

In your model file fordward part use.

```python
output, net_max, net_mean = self.OD(input_image = x)
# x is point cloud data, shape like [B, C, P].
```

## Murmur

這篇我是根據原著的 Quick Start 改的，

基本上就是改寫這份檔案 [network architecture file](https://github.com/samaonline/spatial-transformer-for-3d-point-clouds/blob/master/point_based/part_seg/part_seg_model_deform.py#L53) 第 15 行 到 53 行的功能。 


## Contact

Further information please contact me.

wuyiulin@gmail.com