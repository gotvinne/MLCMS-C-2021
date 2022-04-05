
from objects.dijkstra_table import REACHED_TARGET, DijkstraTable
from objects.transition import Transition

from tkinter import ttk
from tkinter import *

from prettytable import PrettyTable

class Grid(): 
    """GUI object displaying Tables simulation with use of python's tkinter lib
    """
    def __init__(self,simulation_dict,simulation_steps):
        """ Displaying frame determined by simulation_dict

        Args:
            simulation_dict (dictionary): dictionary determing a initial simulation layout
            simulation_steps (n): Total number of simulations steps
        """
        self.simulation_steps = simulation_steps
        self.current_simulation_step = 0

       # NB! Dimentions above 200 (n > 200) is discuraged due to visualization and slow GUI
        self.n = self.get_dimension(simulation_dict.get("N")) 
        del simulation_dict["N"]

        self.table = DijkstraTable(self.n,simulation_dict)
        self.start_simulation = self.table.start_simulation 

        self.screen = Tk()
        self.SCREEN_OFFSET = 100
        self.SCREEN_SIZE = self.screen.winfo_screenheight()-self.SCREEN_OFFSET

        if (self.n < 180):
            self.PAD = 200-self.n
        else:
            self.PAD = 10
        self.SQUARE_SIZE = (self.SCREEN_SIZE-2*self.PAD)//self.n
        self.FIELD = self.SQUARE_SIZE*self.n
        self.pause = False
        
    
        self.step_label = None
        self.simulation_step_str = StringVar()
        self.create_window()
        self.create_grid(self.start_simulation)
        self.create_controlplane()
        self.grid.after(1000,self.update_gui)
        self.screen.mainloop()

    def create_window(self):
        self.screen.title("Cellular automata")
        self.screen.geometry(str(self.SCREEN_SIZE)+"x"+str(self.SCREEN_SIZE))
        self.grid = Canvas(self.screen, bg="grey",width=self.FIELD,height=self.FIELD)

    def create_controlplane(self):
        self.step_label = Label(self.screen, textvariable=self.simulation_step_str, bg="white", fg="black")
        self.step_label.pack(side=TOP)
        
        pause_button = Button(self.screen, text="PAUSE", command=self.pause_simulation) 
        pause_button.pack(side=TOP)
          
        resume_button = Button(self.screen, text="RESUME", command=self.resume_simulation)
        resume_button.pack(side=TOP)   

    def create_grid(self,simulation):
        self.canvas_grid = []
        for i in range(self.n):
            canvas_row = []
            for j in range(self.n):
                if (simulation[i][j].get_state() == "P"):
                    cell = self.grid.create_rectangle(self.SQUARE_SIZE*j,self.SQUARE_SIZE*i,self.SQUARE_SIZE+(self.SQUARE_SIZE*j),self.SQUARE_SIZE+self.SQUARE_SIZE*i,fill="red",outline="black")
                elif (simulation[i][j].get_state() == "E"):
                    cell = self.grid.create_rectangle(self.SQUARE_SIZE*j,self.SQUARE_SIZE*i,self.SQUARE_SIZE+(self.SQUARE_SIZE*j),self.SQUARE_SIZE +self.SQUARE_SIZE*i,fill="white",outline="black")
                elif (simulation[i][j].get_state() == "T"):
                    cell = self.grid.create_rectangle(self.SQUARE_SIZE*j,self.SQUARE_SIZE*i,self.SQUARE_SIZE+(self.SQUARE_SIZE*j),self.SQUARE_SIZE+self.SQUARE_SIZE*i,fill="yellow",outline="black")
                elif (simulation[i][j].get_state() == "O"):
                    cell = self.grid.create_rectangle(self.SQUARE_SIZE*j,self.SQUARE_SIZE*i,self.SQUARE_SIZE+(self.SQUARE_SIZE*j),self.SQUARE_SIZE+self.SQUARE_SIZE*i,fill="black",outline="black")
                canvas_row.append(cell)
            self.canvas_grid.append(canvas_row)
        self.grid.pack()
        
    def update_grid(self):
        """Update the GUI grid determined by a list of transitions
        """
        transitions = self.table.update_simulation()

        for tr in transitions:
            self.grid.itemconfig(self.canvas_grid[tr.old[0]][tr.old[1]],fill="white")
            if (tr.new != REACHED_TARGET):
                self.grid.itemconfig(self.canvas_grid[tr.new[0]][tr.new[1]],fill="red")
        self.grid.pack()
        
    def update_simulation_step(self):
        self.current_simulation_step += 1
        self.simulation_step_str.set("Simulation step:" + str(self.current_simulation_step))
        self.step_label.pack()

    def update_gui(self):
        """Widget interactions and finisching simulation routine
        """
        if (not self.pause):
            if (self.simulation_steps >= 1):
                self.simulation_steps -= 1
                self.update_grid()
                self.update_simulation_step()
                self.grid.after(20,self.update_gui)

        if (self.table.done_simulation() and self.simulation_steps==0):
            finished_ped = self.table.get_finished()
            table = PrettyTable(["Pedestrian position","Speed","Initial distance","Simulation steps"])
            for ped in finished_ped:
                table.add_row(ped.get_results())
            print(table)
            self.screen.destroy()

    def get_dimension(self,dimension_lst):
        if (len(dimension_lst) != 1):
            raise ValueError("Inconsistent dimension")
        return int(dimension_lst[0])
    
    def pause_simulation(self):
        self.pause = True
        
    def resume_simulation(self):
        self.pause = False
        self.grid.after(1000,self.update_gui)