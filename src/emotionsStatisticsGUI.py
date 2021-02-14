from __future__ import unicode_literals
import sys
import os
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets, QtGui

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import pickle

progname = os.path.basename(sys.argv[0])
progversion = "0.1"

      

class MyPieChart(FigureCanvas):
    """Simple canvas with a sine plot."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100, input_data = None):
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(0.1, 0.5, 0.4, 0.6)
        self.axes = fig.add_subplot(111)
        
        emotions_df = self.get_data_from_dict(input_data).sort_values(ascending = False)
        self.emotions = emotions_df.index.to_list()
        self.pie_pieces_sizes = emotions_df.values

        self.compute_initial_figure(self.emotions,self.pie_pieces_sizes)
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        fig.tight_layout()
        
    
    def compute_initial_figure(self, emotions, sizes):
        explode = [0.1]*len(emotions)
        self.axes.pie(sizes,labels=emotions, pctdistance=0.7, autopct='%1.0f%%', 
                      explode = explode ,startangle = 50, labeldistance=1.2)
        
        #draw circle
        centre_circle = plt.Circle((0,0),0.50,fc='white')
        self.axes.add_artist(centre_circle)
        self.axes.set_title('Total emotional condition \n of the group', fontweight="bold",
                            pad = 0, color = 'green', fontsize= 10)
    
    def get_data_from_dict(self, data_dict):
        group_emotions_sum = None
        for student_name in data_dict:           
            
            emotions_list = data_dict[student_name][1].columns.values.tolist()[1:]
            total_student_emotions = data_dict[student_name][1].sum()[emotions_list]
            
            if group_emotions_sum is None:
                group_emotions_sum = total_student_emotions.copy()
            else:
                group_emotions_sum +=  total_student_emotions
            
        pie_pieces_sizes = group_emotions_sum/group_emotions_sum.sum()
        
        return pie_pieces_sizes
                
    
    def update_figure(self, emotions, sizes):
        
        self.axes.cla()
        self.emotions = emotions
        self.sizes = sizes
        self.compute_initial_figure(emotions, sizes)
        self.draw()
        
  
        
        
        
class MyTimeChart(FigureCanvas):
    """Simple canvas with a sine plot."""
    
    def __init__(self, parent=None,width=5, height=4, dpi=100, input_data = None):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(0.1, 0.25, 0.78, 0.8)
        self.axes = self.fig.add_subplot(111)
        
        self.df_data = self.get_data_from_dict(input_data)
        
        # self.emo_time = emo_time
        # self.time = time
        self.compute_initial_figure(self.df_data)
        
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        


    def compute_initial_figure(self,input_data):
        df_data = input_data.copy()
        df_data  = df_data.rolling(40, win_type='gaussian').mean(std = 6)
        df_data.set_index((df_data.index-df_data.index[-1])/60, inplace = True)
        df_data.plot(ax = self.axes)
        # for emotion in emo_time:
        #     self.axes.plot(time, emo_time[emotion])
        
        
        self.axes.set_ylabel('Share of emotion, %', fontweight="bold")
        self.axes.set_xlabel('Time, min', fontweight="bold")
        self.axes.legend(list(df_data.head()), loc=(1.05,0.01),  fontsize= 9)
        self.axes.set_title('Evolution of emotional condition of the group', fontweight="bold",
                            pad =10, color = 'green', fontsize= 10)
        
        
    def get_data_from_dict(self, data_dict):
        group_emotions_sum = None
        
        for student_name in data_dict:           
            
            emotions_df = data_dict[student_name][1].set_index('time')
            emotions_df.replace(np.nan, 0 , inplace = True )

            
            if group_emotions_sum is None:
                group_emotions_sum = emotions_df.copy()
            else:
                 group_emotions_sum +=  emotions_df
                
        pie_pieces_sizes =  group_emotions_sum.div(group_emotions_sum.sum(axis=1)+0.0001, axis=0)
        
        
        return pie_pieces_sizes*100 # Multiplication by 100 needed to convert values to %
        
        

       
            
        

class MyBarChart(FigureCanvas):
    """Simple canvas with a sine plot."""
    
    def __init__(self, parent=None, width=2, height=4, dpi=100, name ='None', input_data = None):
        
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(0.3, 0.3, 0.9, 0.8)
        #fig.tight_layout()
        self.name = name
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure(input_data, self.name)
        

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        


    def compute_initial_figure(self, input_dataframe, name ):

        emotions_df = input_dataframe.set_index('time')
        emotions_df.replace(np.nan, 0 , inplace = True )
        output_df = emotions_df.sum()/emotions_df.sum().sum()*100
        
        
        
      
        self.y_pos = np.arange(len(output_df.index))
        
        self.axes.barh(self.y_pos, output_df.values, align='center', alpha=0.5 ,tick_label = output_df.index)
        

        
        self.axes.set_ylabel('Emotion', fontweight="bold", fontsize= 9)
        self.axes.set_xlabel('Share of emotion per lesson, %', fontweight="bold", fontsize= 9)
        self.axes.set_title(name+'\'s impression of the lesson', fontweight="bold",
                            pad =10, color = 'green', fontsize= 10)
        
        
        
        
    def update_figure(self,input_dataframe, name):
        
        self.axes.cla()
        
        self.compute_initial_figure(input_dataframe, name)
        self.draw()



class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self,data_dict):
        
        ## Get initial statistics data
                
        self.data_dict = data_dict
        
        ######
        
        
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Emotion map of the lesson")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)
        
              
        

        # l1 = QtWidgets.QVBoxLayout(self.main_widget)
        self.l = QtWidgets.QGridLayout(self.main_widget)
        
        
        
        
        self.pie = MyPieChart(self.main_widget, width=2.8, height=2.8, dpi=100, input_data = self.data_dict)
        self.t_graph = MyTimeChart(self.main_widget, width=6, height=6, dpi=100,input_data = self.data_dict)
        
        #self.bar_charts = {}
        #for name in self.emo_name_emo:
        #    bars = self.emo_name_emo[name]
        #    self.bar_charts[name] = MyBarChart(self.main_widget, emotions=self.emotions, numbers = bars, width=3, height=3, dpi=100)
        
        
        
        self.combo = QtWidgets.QComboBox(self.main_widget)
        self.combo.setFont( QtGui.QFont('Arial', 15))
        
        
        
        for name in self.data_dict:
            self.combo.addItem(name)
        
        
        
        
        
        text = QtWidgets.QLabel(self.main_widget)
        text.setText('Choose name of the pupul:')
        text.setFont( QtGui.QFont('Arial', 15))





        self.combo.activated[str].connect(self.onChanged)   
        
                
        
        self.l.addWidget(text,0,0)
        self.l.addWidget(self.combo,0,1)
        self.l.addWidget(self.pie,1,0)
        self.l.addWidget(self.t_graph,2,0,2,0)
        
        bars_daraframe = self.data_dict[self.combo.currentText()][1]
        self.emo_bars1= MyBarChart(self.main_widget, name =self.combo.currentText()
                                   ,input_data = bars_daraframe , 
                                   width=4, height=4, dpi=100)
        self.l.addWidget(self.emo_bars1,1,1)
        
        
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        
        self.statusBar().showMessage("All hail matplotlib!", 2000)
        
        # Show
        self.show()

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """A program for visualizing the emotional
state of schoolchildren during the lesson. Version 0.0.1"""
                                )
    def onChanged(self,name):
        #emo_bars1 = self.bar_charts[text]
        #self.l.addWidget(emo_bars1,1,1)
        
        bars1 = self.data_dict[name][1]
        self.emo_bars1.update_figure(bars1, name)
        
        self.statusBar().showMessage("Changing pupils statictics to "+name + "  ", 2000)
        #self.l.adjustSize()
    
    def update_graphs(self, emotions, time, emo_time,emo_pie_sizes, emo_name_emo ):
        
         ## Get initial statistics data
        self.emotions = emotions
        self.time = time
        self.emo_time = emo_time
        self.emo_pie_sizes = emo_pie_sizes
        self.emo_name_emo = emo_name_emo 
        self.pupils_names = self.emo_name_emo.keys()
        
        ###### Update bars graph
        
        bars1 = self.emo_name_emo[self.combo.currentText()]
        self.emo_bars1.update_figure(bars1, self.combo.currentText())
        
        ###### Update time graph
        
        self.pie.update_figure(self.emotions, self.emo_pie_sizes)
        
        ###### Update time graph
        
        self.t_graph.update_figure(self.time, self.emo_time)

def get_data_from_file():
    
    with open('C:\\Users\\Larionov\\Downloads\\log.dictionary', 'rb') as pickeled_log:
        
        data_dict = pickle.load(pickeled_log)
        
        return  data_dict


def show_statistic_window(logs_dict):
  
    
    qApp = QtWidgets.QApplication(sys.argv)
    
    aw = ApplicationWindow(logs_dict)
    
    aw.setWindowTitle("Emotional detection statistics")

    qApp.exec_()

    
    
        
if __name__ == "__main__":

    
    logs_dict = get_data_from_file()
    
    
    qApp = QtWidgets.QApplication(sys.argv)
    
    aw = ApplicationWindow(logs_dict)
    
    aw.setWindowTitle("Emotional detection statistics")

    qApp.exec_()
