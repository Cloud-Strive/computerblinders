class Array(object):
    def __init__(self, capacity, fillValue = None):
        self.items = list()
        for count in range(capacity):
            self.items.append(fillValue)
def __len__(self):
    """-> The capacity of the array."""
    return len(self.items)
def __str__(self):
    """-> The string representation of the array."""
    return str(self.items)
def __iter__(self):
    """"Supports traversal with a for loop."""
    return iter(self.items)
def __getitem__(self, index):
    """Subscript operator for access at index."""
    return self.items[index]
def __setitem__(self, index, newItem):
    """Subscript operator for replacement at index."""
    self.items[index] = newItem

def sizeIncrease(a):
    if logicalSize == len(a):
        temp = Array(len(a)+1)
        for i in range (logicalSize):
            temp = a[i]
        a = temp
def sizeDecrease(a):
    if logicalSize <= len(a)//4:
        temp = Array(len(a)//2)
        for i in range (logicalSize):
            temp[i] = a[i]
        a = temp