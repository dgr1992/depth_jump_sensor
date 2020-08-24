import numpy as np
import math
import copy
from graphics import GraphWin, Point, Line, Circle, Text

class GapVisualiser:

    def __init__(self, name):
        self.win = GraphWin(name, 500, 500) # give title and dimensions of window
        self.r = 200
        self._setup()
        self.current_gaps = []
        self.current_gaps_nodes = []

    def close(self):
        self.win.close()

    def _setup(self):
        circle = Circle(Point(250,250), self.r) # set center and radius
        circle.setWidth(3)
        circle.draw(self.win)

        line = Line(Point(250,225), Point(250,285))
        line.setWidth(3)
        line.draw(self.win)

        line2 = Line(Point(280,250), Point(250,225))
        line2.setWidth(3)
        line2.draw(self.win)

        line3 = Line(Point(220,250), Point(250,225))
        line3.setWidth(3)
        line3.draw(self.win)

    def _visualise_gap(self, alpha):
        alpha_rad = (alpha * 2 * math.pi / 360) - math.pi / 2
        y = self.r * math.sin(alpha_rad)
        x = -1 * self.r * math.cos(alpha_rad)
        gap = Circle(Point(x + 250, y + 250), 5) # set center and radius
        gap.setFill("red")
        gap.draw(self.win)
        
        return gap

    def _visualise_gap_id(self, alpha, id):
        alpha_rad = (alpha * 2 * math.pi / 360) - math.pi / 2
        y = self.r * math.sin(alpha_rad)
        x = -1 * self.r * math.cos(alpha_rad)

        pos = Point(x + 250, y + 250)

        gap = Circle(pos, 8) # set center and radius
        gap.setFill("red")
        gap.draw(self.win)
        
        text = Text(pos, str(id))
        text.setTextColor("black")
        text.setSize(8)
        text.setStyle('bold')
        text.draw(self.win)

        return (id, gap, text)

    def draw_gaps(self, gaps):
        # first remove old
        if len(self.current_gaps) != 0:
            for gap in self.current_gaps:
                gap.undraw()
        self.current_gaps = []
        for i in range(0, len(gaps)):
            if gaps[i] > 0:
                gap = self._visualise_gap(i)
                self.current_gaps.append(gap)

    def draw_gaps_tree_nodes(self, gaps):
        if len(self.current_gaps_nodes) == 0:
            self.current_gaps_nodes = [None] * len(gaps)

        # if len(self.current_gaps) != 0:
        #    for (gap, text) in self.current_gaps:
        #        gap.undraw()
        #        text.undraw()

        # self.current_gaps = []
        
        for i in range(0, len(gaps)):
            if gaps[i] != None:
                draw = True
                id = gaps[i].id
                if self.current_gaps_nodes[i] != None:
                    (id_current, gap_current, text_current) = self.current_gaps_nodes[i]
                    if id != id_current:
                        text_current.undraw()
                        gap_current.undraw()
                    else:
                        draw = False
                if draw and not gaps[i].appear:
                    gap = self._visualise_gap_id(i, id)
                    self.current_gaps_nodes[i] = gap
            elif self.current_gaps_nodes[i] != None:
                (id, gap, text) = self.current_gaps_nodes[i]
                text.undraw()
                gap.undraw()
                self.current_gaps_nodes[i] = None

if __name__ == '__main__':
    try:
        visualise = GapVisualiser('gap')
        visualise._visualise_gap(90)
    except Exception as ex:
        print(ex)