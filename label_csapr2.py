#to do list:
#   keep masks and radar data on seperate canvases or layers so they can be updated seperately
#   check screen res and start at a reasonable resolution
#   add a paintbrush tool
#   add display of existing raster masks
#   flood-fill tool

import configparser
import json
import tkinter
from tkinter import StringVar, DoubleVar, ttk
from glob import glob1
from netCDF4 import Dataset
import numpy as np
import os
from matplotlib.backends.backend_tkagg import  FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector, LassoSelector
from matplotlib.patches import PathPatch
import pickle
from functools import partial
from utils import PolyMask
import warnings
warnings.filterwarnings('ignore')

#set constants
DEFAULT_DATA_PATH = './data'
DEFAULT_MASK_FILE = 'second_trip.msk'
LASSO_TOOL_RES = 25
POINT_SELECT_RADIUS = 20

#read in info about radar fields from config file:
conf = configparser.ConfigParser()
conf.read('radar_vars.ini')
FIELDS = conf._sections
for var in list(FIELDS.keys()):
    FIELDS[var]['clims'] = json.loads(FIELDS[var]['clims'])
    FIELDS[var]['mask_color'] = json.loads(FIELDS[var]['mask_color'])


class SegmentationApp:
    
    def __init__(self,window):
        #sets up GUI with default values
        
        self.buttons = []                                                       #a list containing references to all of the buttons in the GUI
        
        #frames for navigating between different files and setting output file:
        self.data_file_path = StringVar(value=DEFAULT_DATA_PATH)                #where to look for radar data files
        self.mask_file_name = StringVar(value=DEFAULT_MASK_FILE)                #name of file for mask data
        self.current_mask_name = ''
        self.mask_data = {}
        self.data_file_list = []
        self.current_data_file = StringVar(value='')
        self.update_data_file_list()
        self.load_mask_file()
        self.make_file_nav_frames(window)
        
        #frame for navigating between fields and elevation angles in the netCDF files
        self.current_field = StringVar(value=list(FIELDS.keys())[0])
        self.current_elevation = DoubleVar(value=0.0)
        self.elevation_menu = None
        self.make_field_nav_frame(window)
        
        #frame for plotting radar data and masks:
        self.axes = None
        self.canvas = None
        self.make_plotting_frame(window)
        self.update_plot()
        
        #frame containing buttons for selecting masks:
        self.selector = None
        self.active_selector_name = ''
        self.make_mask_tool_frame(window)
        
        
        
        
        
    ###############################################################################################################################
    # subroutines for file navigation:
    ###############################################################################################################################
        
    def make_file_nav_frames(self,window):
        
        #add a frame for selecting the data input directory:
        inp_dir_frame = tkinter.Frame(window,borderwidth=8)
        inp_dir_frame.pack(anchor='nw')
        self.buttons.append(tkinter.Button(inp_dir_frame,width=16,text='Data File Path:',command=self.on_press_data_file_path))
        self.buttons[-1].pack(side=tkinter.LEFT)
        tkinter.Label(inp_dir_frame,textvariable=self.data_file_path).pack(side=tkinter.LEFT)
       
        #add a frame for selecting the mask file:
        mask_file_frame = tkinter.Frame(window,borderwidth=8)
        mask_file_frame.pack(anchor='w')
        self.buttons.append(tkinter.Button(mask_file_frame, width=16,text='Mask File Name:',command=self.on_press_mask_file_name))
        self.buttons[-1].pack(side=tkinter.LEFT)
        tkinter.Label(mask_file_frame,textvariable=self.mask_file_name).pack(side=tkinter.LEFT)
        
        ttk.Separator(window,orient='horizontal').pack(fill='x')
        
        #buttons for file navigation:
        file_nav_frame = tkinter.Frame(window,borderwidth=4)
        file_nav_frame.pack(anchor='s',side='bottom')
        
        cur_file_frame = tkinter.Frame(file_nav_frame)
        cur_file_frame.pack()
        tkinter.Label(cur_file_frame,text='Current File: ').pack(side=tkinter.LEFT)
        tkinter.Label(cur_file_frame,textvariable=self.current_data_file).pack(side=tkinter.LEFT)
        
        fb_button_frame = tkinter.Frame(file_nav_frame)
        fb_button_frame.pack()
        self.buttons.append(tkinter.Button(fb_button_frame,text='<--Last Masked Case    ',command=partial(self.on_press_skip_to_masked,-1)))
        self.buttons[-1].pack(side=tkinter.LEFT)
        button_deltas = [-10000000,-100,-10,-1,1,10,100,10000000]
        button_labels = ['|<','<100','<10','<','>','10>','100>','>|']
        for delta,label in zip(button_deltas,button_labels):
            self.buttons.append(tkinter.Button(fb_button_frame,text=label,width=4,command=partial(self.on_press_change_file,delta)))
            self.buttons[-1].pack(side=tkinter.LEFT)
        self.buttons.append(tkinter.Button(fb_button_frame,text='    Next Masked Case-->',command=partial(self.on_press_skip_to_masked,1)))
        self.buttons[-1].pack(side=tkinter.LEFT)
            
        ttk.Separator(window,orient='horizontal').pack(fill='x',side='bottom')
        
    
    def on_press_data_file_path(self):
        #Asks user to browse for a directory containing CSAPR2 PPI scans
        new_dir = tkinter.filedialog.askdirectory(initialdir=os.getcwd())
        if not new_dir == '':
            self.data_file_path.set(new_dir)
            self.update_data_file_list()
            self.update_plot()
            
    def on_press_mask_file_name(self):
        #Asks user to type name of a mask file
        new_fname = tkinter.simpledialog.askstring(prompt='Enter Mask Name:',title='Change Mask File')
        if not new_fname is None:
            if new_fname[-5:] != '.poly':
                new_fname += '.poly'
            self.mask_file_name.set(new_fname)
            self.load_mask_file()
            self.update_plot()
    
    def on_press_change_file(self,delta):
        #called when a button is pressed to change the current NetCDF file
        if len(self.data_file_list) > 0:
            file_ind = self.data_file_list.index(self.current_data_file.get())
            new_ind = np.clip(file_ind+delta,0,len(self.data_file_list)-1)
            self.current_data_file.set(self.data_file_list[new_ind])
            if file_ind != new_ind:
                self.update_plot()
        
    def on_press_skip_to_masked(self,delta):
        #skips to the next file/elevation with a mask defined
        mask_names = list(self.mask_data.keys())
        cur_fname = self.current_data_file.get().split('.nc')[0]
        if len(mask_names)==0:
            return
        if not self.current_mask_name in mask_names:
            mask_names.append(cur_fname)
        else:
            cur_fname = self.current_mask_name
        mask_names.sort()
        new_idx = np.clip(mask_names.index(cur_fname)+delta,0,len(mask_names)-1)
        new_fname = mask_names[new_idx].split('_elv=')
        self.current_data_file.set(new_fname[0]+ '.nc')
        self.current_elevation.set(new_fname[1])
        self.update_plot()
    
    
    ###############################################################################################################################        
    #subroutines for loading and saving data:
    ###############################################################################################################################
            
    def update_data_file_list(self):
        self.data_file_list = glob1(self.data_file_path.get(),'*.nc')
        self.data_file_list.sort()
        if len(self.data_file_list)==0:
            self.current_data_file.set('')
            print('No netCDF files found in selected path')
        else:
            self.current_data_file.set(self.data_file_list[0])
            print('Found ' + str(len(self.data_file_list)) + ' netCDF files')
    
    def load_mask_file(self):
        if os.path.exists('./' + self.mask_file_name.get()):
            self.mask_data = pickle.load(open(self.mask_file_name.get(),'rb'))
            print('Loaded existing mask file with ' + str(len(self.mask_data.keys())) + ' entries: ' + self.mask_file_name.get())
        else:
            self.mask_data = {}
            print('Created new mask file: ' + self.mask_file_name.get())
            self.save_mask_data()
            
    def save_mask_data(self):
        pickle.dump(self.mask_data,open(self.mask_file_name.get(),'wb'))
        print('Wrote mask file with ' + str(len(self.mask_data.keys())) + ' entries: ' + self.mask_file_name.get())
        
        
        
    ###############################################################################################################################
    #FIELD AND ELEVATION PANE:
    ###############################################################################################################################
    def make_field_nav_frame(self,window):
        
        #buttons for selecting field and elevation angle:
        field_nav_frame = tkinter.Frame(window,borderwidth=4)
        field_nav_frame.pack(anchor='s',side='bottom')
        
        #dropdown for selecting elevation:
        tkinter.Label(field_nav_frame,text='Elevation Angle:').pack(side=tkinter.LEFT)
        self.elevation_menu = ttk.OptionMenu(field_nav_frame, self.current_elevation, command=self.update_plot)
        self.elevation_menu.pack(side=tkinter.LEFT)
        
        #radio buttons for selecting field
        tkinter.Label(field_nav_frame,text='    Select Field:').pack(side=tkinter.LEFT)
        for value in FIELDS.keys():
            self.buttons.append(tkinter.Radiobutton(field_nav_frame, text=FIELDS[value]['display_name'], variable=self.current_field, value=value,command=self.update_plot))
            self.buttons[-1].pack(side=tkinter.LEFT)
        
        ttk.Separator(window,orient='horizontal').pack(fill='x',side='bottom')
    
    
    
    ###############################################################################################################################
    #PLOTTING PANE
    ###############################################################################################################################
    def make_plotting_frame(self,window):
        fig = Figure(figsize=(8, 8), dpi=100)
        self.axes = fig.add_axes([0,0,1,1])
        self.axes.axis('off')
        self.canvas = FigureCanvasTkAgg(fig, master=window)
        self.canvas.get_tk_widget().pack(anchor='w',side=tkinter.LEFT)
    
    def update_plot(self,new_elevation=None):
        #called when target file/field/elevation angle changes:
            
        #check that the currently selected file exists
        fpath = os.path.join(self.data_file_path.get(),self.current_data_file.get())
        if not os.path.isfile(fpath):
            print('File not found! ' + fpath)
            return
        
        #read in data:
        ncf = Dataset(fpath)
        elv = list(np.round(ncf.variables['fixed_angle'][:],1))
        
        #update the dropdown menu with elevation angles:
        #(if the currently set elevation is in the new list don't change it)
        if np.float32(self.current_elevation.get()) in elv:
            self.elevation_menu.set_menu(float(self.current_elevation.get()),*elv)
        else:
            self.elevation_menu.set_menu(elv[0],*elv)
        
        #load in and plot the data for the current field:
        start_ray_idx = ncf.variables['sweep_start_ray_index'][:][elv.index(np.float32(self.current_elevation.get()))]
        start_azm = ncf.variables['azimuth'][:][start_ray_idx]
        data = ncf.variables[FIELDS[self.current_field.get()]['netcdf_name']][:][start_ray_idx:start_ray_idx+360,:]
        t = np.pi*(np.linspace(0,360,data.shape[0])+start_azm)/180
        r = np.arange(0,data.shape[1])
        [r,t] = np.meshgrid(r,t)
        x, y = np.cos(t)*r, np.sin(t)*r
        clim = FIELDS[self.current_field.get()]['clims']
        cmap = FIELDS[self.current_field.get()]['colormap']
        self.axes.clear()
        self.axes.pcolormesh(x,y,data,vmin=clim[0],vmax=clim[1],shading='nearest',cmap=cmap)
        self.axes.axis('off')
        self.draw_mask()
        self.canvas.draw()
        
    def draw_mask(self):
        self.axes.patches=[]
        #plot the mask:
        self.current_mask_name = self.current_data_file.get()[:-3] + '_elv=' + str(self.current_elevation.get())
        if self.current_mask_name in self.mask_data:
            paths = self.mask_data[self.current_mask_name].as_paths()
            if len(paths) == 0:
                self.axes.text(0,0,'Empty Mask',fontsize=20,fontweight='bold',backgroundcolor=[1,1,1,0.5],ha='center',va='center')
            for path in paths:
                new_patch = PathPatch(path,facecolor=FIELDS[self.current_field.get()]['mask_color'])
                self.axes.add_patch(new_patch)
        self.canvas.draw()
                
                
                
                
                
                
    ###############################################################################################################################
    #MASK SELECTION FUNCTIONS:
    ###############################################################################################################################
    def make_mask_tool_frame(self,window):
        
        #buttons for selecting masked regions:
        mask_menu_frame = tkinter.Frame(window,borderwidth=4)
        mask_menu_frame.pack(expand=True,side=tkinter.LEFT)
        
        tkinter.Label(mask_menu_frame,text='Mask Regions:').pack()
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Polygon',width=16,command=self.on_press_select_polygon))
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Lasso',width=16,command=self.on_press_select_lasso))
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Point',width=16,command=self.on_press_select_point))
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Delete Region',width=16,command=self.on_press_delete))
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Delete Last',width=16,command=self.on_press_delete_last))
        for i in range(-5,0): self.buttons[i].pack();
        tkinter.Button(mask_menu_frame,text='Done Selecting',width=16,command=self.on_press_done_selecting).pack()
        
        tkinter.Label(mask_menu_frame,text='           ').pack()
        tkinter.Label(mask_menu_frame,text='Mask Menu:').pack()
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Mark as Empty',width=16,command=self.on_press_mark_as_empty))
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Refresh Plot',width=16,command=self.update_plot))
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Merge Polygons',width=16,command=self.on_press_merge_polygons))
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Clear Mask',width=16,command=self.on_press_clear_mask))
        for i in range(-4,0): self.buttons[i].pack();
        
        tkinter.Label(mask_menu_frame,text='           ').pack()
        tkinter.Label(mask_menu_frame,text='File Menu:').pack()
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Save Mask File',width=16,command=self.save_mask_data))
        self.buttons.append(tkinter.Button(mask_menu_frame,text='Quit',width=16,command=window.quit))
        for i in range(-2,0): self.buttons[i].pack();
    
    def disable_buttons(self):
        for b in self.buttons:
            b['state'] = 'disable'
        self.elevation_menu.config(state='disable')
        
    def on_press_select_polygon(self):
        self.disable_buttons()
        self.active_selector_name = 'polygon'
        self.selector = PolygonSelector(self.axes,self.on_select_polygon)
    
    def on_select_polygon(self,verts):
        self.add_mask(verts)
        self.selector.disconnect_events()
        self.selector = PolygonSelector(self.axes,self.on_select_polygon)
    
    def on_press_select_lasso(self):
        self.disable_buttons()
        self.active_selector_name = 'lasso'
        self.selector = LassoSelector(self.axes,self.on_select_lasso)
    
    def on_select_lasso(self,verts):
        
        #coarsen the lasso data
        coarse_verts = [verts[0]]
        for i in range(1,len(verts)):
            dist = np.sqrt((coarse_verts[-1][0]-verts[i][0])**2 + (coarse_verts[-1][1]-verts[i][1])**2)
            if dist > LASSO_TOOL_RES:
                coarse_verts.append(verts[i+1])
        if coarse_verts[-1] != verts[-1]:
            coarse_verts.append(verts[-1])
        
        #add the lasso selection
        self.add_mask(coarse_verts)
        self.selector.disconnect_events()
        self.selector = LassoSelector(self.axes,self.on_select_lasso)
    
    def on_press_select_point(self):
        self.disable_buttons()
        self.active_selector_name = 'point'
        self.selector = self.canvas.mpl_connect('button_press_event',self.on_point_select)
        
    def on_point_select(self,event):
        verts = []
        x0,y0 = event.xdata, event.ydata
        for theta in np.arange(0,2*np.pi,np.pi/8):
            x = np.cos(theta)*POINT_SELECT_RADIUS
            y = np.sin(theta)*POINT_SELECT_RADIUS
            verts.append((x0+x,y0+y))
        self.add_mask(verts)
        
    def on_press_delete(self):
        self.disable_buttons()
        self.active_selector_name = 'delete'
        self.selector = self.canvas.mpl_connect('button_press_event',self.on_point_delete)
        
    def on_point_delete(self,event):
        if self.current_mask_name in self.mask_data:
            self.mask_data[self.current_mask_name].delete_point(event.xdata,event.ydata)
        self.draw_mask()
        
    def on_press_mark_as_empty(self):
        self.mask_data[self.current_mask_name] = PolyMask()
        self.draw_mask()
    
    def on_press_done_selecting(self):
        if self.active_selector_name == 'polygon':
            self.selector.disconnect_events()
        if self.active_selector_name == 'lasso':
            self.selector.disconnect_events()
        if self.active_selector_name == 'point':
            self.canvas.mpl_disconnect(self.selector)
        if self.active_selector_name == 'delete':
            self.canvas.mpl_disconnect(self.selector)
        self.selector = None
        self.active_selector_name = ''
        for b in self.buttons:
            b['state'] = 'normal'
        self.elevation_menu.configure(state='normal')
        self.update_plot()
        
    def on_press_delete_last(self):
        if self.current_mask_name in self.mask_data:
            self.mask_data[self.current_mask_name].pop()
            self.update_plot()
    
    def on_press_clear_mask(self):
        if self.current_mask_name in self.mask_data:
            self.mask_data.pop(self.current_mask_name)
            self.update_plot()
    
    def on_press_merge_polygons(self):
        if self.current_mask_name in self.mask_data:
            self.mask_data[self.current_mask_name].union()
        self.draw_mask()
    
    def on_press_export_masks(self):
        #converts the mask polygons to raster data in polar coordinates:
        print('this function doesn''t do anything yet')
        
    def add_mask(self,verts):
        if self.current_mask_name in self.mask_data:
            self.mask_data[self.current_mask_name].add(verts)
        else:
            self.mask_data[self.current_mask_name] = PolyMask()
            self.mask_data[self.current_mask_name].add(verts)
        self.draw_mask()
        
        
#create the window
window = tkinter.Tk()
window.geometry('1000x1000+10+10')
window.title('CSAPR2 Labeling')
SegmentationApp(window)
window.mainloop()
