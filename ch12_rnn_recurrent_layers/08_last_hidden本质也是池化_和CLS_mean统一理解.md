# `last hidden` 本质也是池化：和 CLS / mean 的统一理解

这篇只解决一个常见混淆：  
**RNN 里“取最后一个 hidden”到底算不算池化？**

---

## 1) 先给结论

算，而且就是一种最简单的池化策略。

因为池化本质定义只有一个：
- 输入：一串向量
- 输出：一个向量

只要完成这个“序列 -> 单向量”的压缩动作，就叫池化。

---

## 2) 为什么 `last hidden` 也叫池化

RNN 编码后得到的是一串隐藏状态：

`[h_1, h_2, ..., h_T]`

你如果只取最后一个：

`sentence_vector = h_T`

这等价于一种池化策略：
- 策略名可理解为 `last pooling`
- 动作是“只保留最后一个，丢弃前面全部”

---

## 3) 把你常见的写法对齐成同一模板

你以前熟悉的简写：

`RNN -> last hidden -> 句子向量`

更完整的写法：

`RNN -> 序列隐藏状态 -> pooling(last) -> 句子向量`

两者没有矛盾，只是第二种把中间动作点名了。

---

## 4) 那为什么又出现 mean / max / CLS

因为 `last hidden` 有局限：
- 对长序列前部信息不够稳
- 太依赖最后时刻状态质量

所以会有更稳的池化策略：
- `mean pooling`：全序列平均
- `max pooling`：逐维取最大
- `CLS pooling`：取特殊位置向量（Transformer 常见）

它们和 `last hidden` 本质一样，都是 pooling，只是策略不同。

---

## 5) RNN 与 Transformer 的统一公式

统一模板始终是：

`序列编码器(RNN/Transformer) -> pooling策略(last/mean/max/CLS) -> 句子向量`

所以不是“RNN 不需要池化”，而是“RNN 常常默认用了 last pooling”。

---

## 6) 放到 RAG 里为什么必须做这一步

向量库通常存的是“每段文本一个固定向量”，而不是一串 token 向量。  
因此无论编码器是谁，最后都要做一次 pooling 得到单向量表示。

---

## 最简背诵版

1. `last hidden` 本质就是一种 pooling（last pooling）  
2. pooling 的本质是“序列向量 -> 单个向量”  
3. `last / mean / max / CLS` 只是不同 pooling 策略  
4. RNN 与 Transformer 在“编码后要池化”这一点上完全一致
