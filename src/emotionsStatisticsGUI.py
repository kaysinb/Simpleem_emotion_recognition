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
#############
import multiprocessing
#############

progname = os.path.basename(sys.argv[0])
progversion = "0.2"
SAVE_FIG_PATH = ''
      
class MyPieChart(FigureCanvas):
    """ 
        Pie chart class. 
    """
        
    def __init__(self, parent=None, width=5, height=4, dpi=300, input_data = None):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(0.1, 0.5, 0.4, 0.6)
        self.axes = self.fig.add_subplot(111)
        
        emotions_df = self.get_data_from_dict(input_data).sort_values(ascending = False)
        self.emotions = emotions_df.index.to_list()
        self.pie_pieces_sizes = emotions_df.values
        #self.axes.set_xlim(0)

        self.compute_initial_figure(self.emotions,self.pie_pieces_sizes)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        self.fig.tight_layout()
        
    
    def compute_initial_figure(self, emotions, sizes):
        global emotion_color_dict
        explode = [0.1]*len(emotions)
        em_colors = []
        for emotion in emotions:
            em_colors.append(emotion_color_dict[emotion])
        self.axes.pie(sizes,labels=emotions, colors = em_colors, pctdistance=0.7, autopct='%1.0f%%', 
                      explode = explode, startangle = 20, labeldistance=1.2)
        
        self.axes.set_title('Total emotional condition \n of the group', fontweight="bold",
                            pad = 0, color = 'green', fontsize= 10)
        #draw circle
        centre_circle = plt.Circle((0,0),0.50,fc='white')
        self.axes.add_artist(centre_circle)
        
    
    def get_data_from_dict(self, data_dict):
        group_emotions_sum = None
        for student_name in data_dict:           
            
            
            total_student_emotions = data_dict[student_name].sum(axis = 0)
            
            if group_emotions_sum is None:
                group_emotions_sum = total_student_emotions.copy()
            else:
                group_emotions_sum +=  total_student_emotions
            
        pie_pieces_sizes = group_emotions_sum/group_emotions_sum.sum()
        pie_pieces_sizes = pie_pieces_sizes.replace(0, np.nan)
        pie_pieces_sizes = pie_pieces_sizes.dropna(how='all', axis=0)
        
        return pie_pieces_sizes
                
    
    def update_figure(self):
        
        global using_data

        self.axes.cla()
        df_data = self.get_data_from_dict(using_data).sort_values(ascending = False)
        self.emotions = df_data.index.to_list()
        self.pie_pieces_sizes = df_data.values
        self.compute_initial_figure(self.emotions,self.pie_pieces_sizes)
        self.draw()
    
    def save_to_file(self, path = 'PieChart.jpg'):
        self.fig.savefig(SAVE_FIG_PATH + path)
        
 
class MyTimeChart(FigureCanvas):
    """Simple canvas with a sine plot."""
    
    def __init__(self, parent=None,width=5, height=4, dpi=100, input_data = None):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(0.1, 0.25, 0.78, 0.8)
        self.axes = self.fig.add_subplot(111)
        
        self.df_data = self.get_data_from_dict(input_data)
        

        self.compute_initial_figure(self.df_data)
        
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        


    def compute_initial_figure(self,input_data):
        global emotion_color_dict
        df_data = input_data.copy()
        em_colors = []
        for emotion in list(df_data.head()):
            em_colors.append(emotion_color_dict[emotion])      
        
        df_data  = df_data.rolling(len(df_data)//50+1, win_type='gaussian').mean(std = 10)
        
        self._line_ = df_data.plot(ax = self.axes, use_index = True, color = em_colors)

        self.axes.set_ylabel('Share of emotion, %', fontweight="bold")
        self.axes.set_xlabel('Time, min', fontweight="bold")
        self.axes.set_xlim(0,None)
        self.axes.set_ylim(0,100)
        self.axes.legend(list(df_data.head()), loc=(1.05,0.01),  fontsize= 10, edgecolor = 'white')
        self.axes.set_title('Evolution of emotional condition of the group', fontweight="bold",
                            pad =10, color = 'green', fontsize= 10)
        
        
    def get_data_from_dict(self, data_dict):
        group_emotions_sum = None
        
        for student_name in data_dict:           
            
            emotions_df = data_dict[student_name]
            
            if group_emotions_sum is None:
                group_emotions_sum = emotions_df.copy()
            else:
                 group_emotions_sum +=  emotions_df
                
        pie_pieces_sizes =  group_emotions_sum.div(group_emotions_sum.sum(axis=1)+0.0001, axis=0)
        
        
        return pie_pieces_sizes*100 # Multiplication by 100 needed to convert values to %
    
    def update_figure(self):
            global using_data


            self.axes.cla()
            self.df_data = self.get_data_from_dict(using_data)
            self.df_data.set_index((self.df_data.index - self.df_data.index[0])/60, inplace = True)
            self.compute_initial_figure(self.df_data)
            self.draw()
    
    def save_to_file(self, path = 'TimeChart.jpg'):
        self.fig.savefig(SAVE_FIG_PATH + path)

        

class MyBarChart(FigureCanvas):
    """Bar chart class"""
    
    def __init__(self, parent=None, width=2, height=4, dpi=100, name ='None', input_data = None):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(0.3, 0.3, 0.9, 0.8)
        #fig.tight_layout()
        self.name = name
        self.df_data = input_data
        self.axes = self.fig.add_subplot(111)

        self.compute_initial_figure(self.df_data, self.name)
  
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
               


    def compute_initial_figure(self, input_dataframe, name ):

        global emotion_color_dict
        output_df =  input_dataframe.sum()/ input_dataframe.sum().sum()*100
        
        em_colors = []
        for emotion in list(output_df.index):
            em_colors.append(emotion_color_dict[emotion])  
        
      
        self.y_pos = np.arange(len(output_df.index))
        
        self.axes.barh(self.y_pos, output_df.values, align='center', alpha=0.5 ,tick_label = output_df.index, color = em_colors)
        self.axes.set_xlim(0,100)       
        self.axes.set_ylabel('Emotion', fontweight="bold", fontsize= 9)
        self.axes.set_xlabel('Share of emotion per lesson, %', fontweight="bold", fontsize= 9)
        self.axes.set_title(name+'\'s impression of the lesson', fontweight="bold",
                            pad =10, color = 'green', fontsize= 10)
        
    def update_figure(self):
        global using_data
        #input_dataframe = get_data_from_queue()
        self.df_data = using_data
        self.axes.cla()
        self.compute_initial_figure(self.df_data[self.name], self.name)
        self.draw()
        
    def save_to_file(self, path = 'TimeChart.jpg'):
        self.fig.savefig(SAVE_FIG_PATH + path)



class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self,data_dict, input_queue):
        
        ## Get initial statistics data
                
        self.data_dict = data_dict
        self.queue = input_queue
        
        ######
        
        
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Emotion map of the lesson")

        self.file_menu = QtWidgets.QMenu('&File', self)
        # self.file_menu.addAction('&Save graph', self.saveGraph,
        #                          QtCore.Qt.CTRL + QtCore.Qt.Key_S)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)
        
        # Graph menu
        self.single_emotion_on = False
        self.graph_menu = QtWidgets.QMenu('&Graph style', self)
        self.graph_menu.addAction('&Use simple statistic', self.single_emotion_style,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_A)
        self.menuBar().addMenu(self.graph_menu)
        ####################################
        
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
        text.setText('Choose name of the student:')
        text.setFont( QtGui.QFont('Arial', 15))

        self.combo.activated[str].connect(self.onChanged)
        self.l.addWidget(text,0,0)
        self.l.addWidget(self.combo,0,1)
        self.l.addWidget(self.pie,1,0)
        self.l.addWidget(self.t_graph,2,0,2,0)
        
        bars_daraframe = self.data_dict[self.combo.currentText()]
        self.emo_bars1= MyBarChart(self.main_widget, name =self.combo.currentText()
                                   ,input_data = bars_daraframe , 
                                   width=4, height=4, dpi=100)
        self.l.addWidget(self.emo_bars1,1,1)
        
        
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        
        self.statusBar().showMessage("All hail matplotlib!", 2000)
        
        # ############################################################
        # Initiate the timer to update graphs
        interval = 200
        self.t_graph._timer_ = self.t_graph.new_timer(interval, [(self.update_graphs, (), {})])
        self.t_graph._timer_.start()
        # ############################################################
        
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
    def single_emotion_style(self):
        
        self.single_emotion_on = not self.single_emotion_on
    
    def onChanged(self,name):
        #emo_bars1 = self.bar_charts[text]
        #self.l.addWidget(emo_bars1,1,1)
        
        bars1 = self.data_dict[name]
        self.emo_bars1.name = name
        self.emo_bars1.df_data = bars1
        self.emo_bars1.update_figure()
        
        self.statusBar().showMessage("Changing pupils statictics to "+name + "  ", 2000)
        #self.l.adjustSize()
    
    def update_graphs(self):
        
        _ = get_data_from_queue(self.single_emotion_on)
        # ###### Update bars graph
        
        self.emo_bars1.update_figure()
        
        # ###### Update time graph
        self.pie.update_figure()
        
        ###### Update time graph
        self.t_graph.update_figure()

def get_data_from_file():
    
    with open('C:\\Users\\Larionov\\Downloads\\log.dictionary', 'rb') as pickeled_log:
        
        data_dict = pickle.load(pickeled_log)
        
        return  data_dict

def log_data_pre_process(input_logs_dict, single_emotion_on):
    emotion_dict = {}
    # Processing log data to left only max value of the emotion type
    for student_name in input_logs_dict:
        emotions_df = input_logs_dict[student_name][1].set_index('time')
        emotions_df.replace(np.nan, 0 , inplace = True )
        if single_emotion_on:
            fl = (emotions_df != 0).any(axis=1)
            emotions_df[fl] = emotions_df[fl].eq(emotions_df[fl].max(axis=1), axis=0).astype(int)
        emotion_dict[student_name] = emotions_df
    
    return emotion_dict
    


# Global variables
input_queue = None  # Queue channel for data input from the mother process
raw_data = None     # Raw input emotional data from the mother process
one_em_data = None  # Data with only one emotion left with maximum share (replaced by 1)
using_data = None   # Data? whic is used for grahs plotting 

# Selected interface colors for emotions
emotion_color_dict = {'happy':(249/255,197/255,70/255),
                      'neutral':(83/255,188/255,151/255),
                      'angry':(236/255,90/255,94/255),
                      'sad':(160/255,223/255,232/255),
                      'scared':(208/255,106/255,204/255)}

def real_time_window(input_channel, start_dict):
    """
        Function which is udes for real time updatable video initialization.
    """
    global input_queue, using_data, raw_data, one_em_data
    # Clobal variables initialization
    input_queue = input_channel
    one_em_data = log_data_pre_process(start_dict, True)
    raw_data = log_data_pre_process(start_dict, False)
    using_data = raw_data
    # Window initialization
    qApp = QtWidgets.QApplication([])
    aw = ApplicationWindow(using_data, input_queue)
    aw.setWindowTitle("Emotional detection statistics")
    qApp.exec_()
    


def get_data_from_queue(single_emotion_on):
    global input_queue, using_data, raw_data, one_em_data
    if not input_queue.empty():
        # Getting log from queue
        one_em_frame_dict = log_data_pre_process(input_queue.get(), True)
        raw_frame_dict = log_data_pre_process(input_queue.get(), False)
        # Adding one frame log to total log
        for student_name in one_em_frame_dict:
            one_em_data[student_name] = pd.concat([one_em_data[student_name], one_em_frame_dict[student_name]])
            raw_data[student_name] =  pd.concat([raw_data[student_name], raw_frame_dict[student_name]])
        ########################
    if single_emotion_on:
        using_data = one_em_data
    else:
        using_data = raw_data
    return using_data



def get_real_time_stat_window(total_class_log):
    """
    Function for initialization of dynamic and interractive emotional statistic window.
    
    Parameters
    ----------
    total_class_log : output of Student.get_group_log or Student.get_frame_log
        DESCRIPTION.

    Returns
    -------
    queue_for_logs : Queue  
        Queue object for statistics data transfer from Student class to 
        new process for .
    stat_window : Process object.

    """
    queue_for_logs = multiprocessing.Queue()
    stat_window =  multiprocessing.Process(target = real_time_window, args = (queue_for_logs,total_class_log ,))
    stat_window.start()
    return queue_for_logs, stat_window
    
    
###############################################################

    
    
        
# if __name__ == "__main__":
    
    
    # logs_dict = get_data_from_file()
    # qApp = QtWidgets.QApplication(sys.argv)
    # aw = ApplicationWindow(logs_dict)
    # aw.setWindowTitle("Emotional detection statistics")
    # qApp.exec_()
