from typing import List, Dict, Tuple, Optional
import numpy as np
from Bio import Nexus
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

class Observation:
    def __init__(self, nexus_data: Optional[str] = None, 
                 alignment: Optional[MultipleSeqAlignment] = None,
                 taxa: Optional[List[str]] = None,
                 sequences: Optional[Dict[str, str]] = None,
                 char_labels: Optional[List[str]] = None):
        """
        Args:
            nexus_data: Nexus格式的字符串数据
            alignment: BioPython的多序列比对对象
            taxa: 分类单元名称列表
            sequences: 字典格式的序列数据 {taxon: sequence}
            char_labels: 特征标签列表（对于形态学数据）
        """
        self.nexus_data = nexus_data
        self.alignment = alignment
        self.taxa = taxa or []
        self.sequences = sequences or {}
        self.char_labels = char_labels or []
        self.sequence_length = 0
        self.num_taxa = 0