## 基本结构
Nexus文件由多个块（Blocks） 组成，每个块包含特定类型的数据：
```
#NEXUS

BEGIN BLOCKNAME;
    [指令和参数]
    [数据矩阵]
END;
```
## 主要块类型
### 1.TAXA块
存储分类单元信息
```
BEGIN TAXA;
    DIMENSIONS NTAX=4;                # 有4个分类单元
    TAXLABELS                          # 分类单元标签
        Human
        Chimpanzee
        Gorilla
        Orangutan
    ;
END;
```
### 2.CHARACTERS块
存储特征数据（序列或形态）
```
BEGIN CHARACTERS;
    DIMENSIONS NCHAR=10;              # 10个特征/位点
    FORMAT DATATYPE=DNA MISSING=? GAP=-; # 数据类型和格式
    MATRIX
        Human       ATGCTAGCTA
        Chimpanzee  ATGCTAGCTG
        Gorilla     ATGCTAGCTC
        Orangutan   ATGTTAGCTA
    ;
END;
```
### 3.TREES块
存储系统发育树
```
BEGIN TREES;
    TREE Tree1 = ((Human,Chimpanzee),Gorilla,Orangutan);
    TREE Tree2 = (Human,(Chimpanzee,(Gorilla,Orangutan)));
END;
```

## 树表示法
Nexus文件中的树使用Newick格式表示：
格式              | 描述
---               | ---
(A,B)             | A和B是姐妹群 (由一个共同的最近祖先演化)
(A,B)C            | A和B组成的分支与C是姐妹群
A:0.1             | 分支长度为0.1
(A:0.1,B:0.2):0.3 | 内部节点也有分支长度
