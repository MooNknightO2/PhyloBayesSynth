from typing import Optional


class Expression:
    pass

class N3(Expression):
    def __init__(self):
        self._type = self.__class__.__name__

    @property
    def type(self):
        return self._type

class CRB(N3):
    def __init__(self, para = None):
        super().__init__()
        self.para = para
    def __repr__(self):
        return f"CRB({self.para})"
    
class CRBD(N3):
    def __init__(self, para1 = None, para2 = None):
        super().__init__()
        self.para1 = para1
        self.para2 = para2
    def __repr__(self):
        return f"CRBD({self.para1}, {self.para2})"

class TDB(N3):
    def __init__(self, para1 = None, para2 = None):
        super().__init__()
        self.para1 = para1
        self.para2 = para2
    def __repr__(self):
        return f"TDB({self.para1}, {self.para2})"

class TDBD(N3):
    def __init__(self, para1 = None, para2 = None, para3 = None, para4 = None):
        super().__init__()
        self.para1 = para1
        self.para2 = para2
        self.para3 = para3
        self.para4 = para4
    def __repr__(self):
        return f"TDBD({self.para1}, {self.para2}, {self.para3}, {self.para4})"
    
class BAMM(N3):
    def __init__(self, para = None, nt1: N3 = None, nt2: N3 = None):
        super().__init__()
        self.para = para
        self.nt1 = nt1
        self.nt2 = nt2
    def __repr__(self):
        return f"BAMM({self.para}, {self.nt1.type}, {self.nt2.type})"
