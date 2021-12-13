class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg: .4f}"
        return text

