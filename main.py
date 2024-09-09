import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as transforms

# ALL Tests should be in the same CSV for this!
# Column order in the output.ipynb must be respected!
class Dataset:
    def __init__(self, path, delimiter, encode, newline, discard_negative_modulus):
        self.path = path
        self.delimiter = delimiter
        self.encode = encode
        self.newline = newline
        self.test_list = []
        self.discard_negative_modulus = discard_negative_modulus

        self.csvobj = self.GetCSVobject()

    # Returns list obj
    def GetCSVobject(self):
        with open(self.path, newline=self.newline, encoding=self.encode) as csvfile:
            csv_obj = list(csv.reader(csvfile, delimiter=self.delimiter))
        return csv_obj

    def ProcedureAntonPaar(self):
        test_list = []

        collect_data_flag = False
        for row in range(len(self.csvobj)):
            curr_line = self.csvobj[row]

            # AD-HOC Excuse for constant component 
            if not(all([el=='' for el in curr_line]) or ('Constant component' in curr_line)):
                # Separating tests
                if 'Test:' in curr_line:
                    collect_data_flag = False
                    curr_interval = -1
                    curr_results = -1
                    curr_instance = Tests(curr_line[1])
                    test_list.append(curr_instance)
                # Separating results
                elif 'Result:' in curr_line:
                    collect_data_flag = False
                    curr_results += 1
                    curr_interval = -1
                    curr_instance.results.append(curr_line[1])
                    curr_instance.data.append([])
                    curr_instance.plottable.append([])
                # Separating intervals
                elif 'Interval and data points:' in curr_line:
                    collect_data_flag = False
                    curr_interval += 1
                    curr_instance.number_of_intervals += 1
                    curr_instance.data[curr_results].append([])
                    curr_instance.plottable[curr_results].append([])
                # Assign variables
                elif 'Interval data:' in curr_line:
                    collect_data_flag = False
                    curr_instance.data_columns = curr_line[2:]
                    for _ in range(len(curr_instance.data_columns)):
                        curr_instance.data[curr_results][curr_interval].append([])
                # Assign units
                elif any([i==j for i in ['[ms]', '[s]', '[min]', '[h]', '[d]'] for j in curr_line]):
                    collect_data_flag = True
                    curr_instance.data_units = curr_line[2:]
                    continue
                # Obtain values
                elif collect_data_flag is True and (('(not used)' not in curr_line[1]) and ('invalid' not in curr_line[1])):
                    for colidx in range(len(curr_line)-2):
                        curr_instance.data[curr_results][curr_interval][colidx].append(float(curr_line[colidx+2]))
                    curr_instance.plottable[curr_results][curr_interval].append('On')
        
        self.test_list = test_list[:]
        if self.discard_negative_modulus:
            self.RemoveNegativeModulus()

    def RemoveNegativeModulus(self):
        var_modulus = self.test_list[0].data_columns.index('Elastic Modulus')
        for test in self.test_list:
            for residx in range(len(test.data)):
                for intidx in range(len(test.data[residx])):
                    for dataidx in range(len(test.data[residx][intidx][var_modulus])):
                        if test.data[residx][intidx][var_modulus][dataidx] < 1:
                            test.plottable[residx][intidx][dataidx] = 'Deactivated'


    def CallTestbyName(self, test):
        test_names = [i.name for i in self.test_list]
        test_idx = test_names.index(test)
        return self.test_list[test_idx]

    def MultipleTestPlot(self, test_names, var1, var2, log=False, one_curve=False):
        called_tests = [self.CallTestbyName(test) for test in test_names]

        fig = plt.figure(figsize=(19.2, 10.8))
        axi = fig.add_subplot(1,1,1)
        axi.set_xlabel(var1, fontsize=20)
        axi.set_ylabel(var2, fontsize=20)
        axi.xaxis.set_tick_params(labelsize=20)
        axi.yaxis.set_tick_params(labelsize=20)

        for test in called_tests:
            varx = test.data_columns.index(var1)
            vary = test.data_columns.index(var2)

            one_curve_plot_x = []
            one_curve_plot_y = []
            for residx in range(len(test.data)):
                for intidx in range(len(test.data[residx])):
                    plotx = []
                    ploty = []
                    # Data to be plotted
                    for elidx in range(len(test.plottable[residx][intidx])):
                        if test.plottable[residx][intidx][elidx] != 'Deactivated' and test.plottable[residx][intidx][elidx] != 'Off':
                            plotx.append(test.data[residx][intidx][varx][elidx])
                            ploty.append(test.data[residx][intidx][vary][elidx])
                    if one_curve:
                        one_curve_plot_x += plotx
                        one_curve_plot_y += ploty 
                    if len(plotx) > 0 and (one_curve is False):
                        axi.plot(plotx, ploty, linewidth=3, 
                                label=test.name+'-R'+str(residx+1)+'I'+str(intidx+1))
            if len(one_curve_plot_x) > 0 and one_curve:
                axi.plot(one_curve_plot_x, one_curve_plot_y, linewidth=3, 
                        label=test.name)

        if log:
            axi.set_yscale('log')
            axi.set_ylim(bottom=1)
        if var2 == 'Elastic Modulus':
            axi.set_yscale('log')
            axi.set_ylim(bottom=1)
        axi.autoscale(True)
        legend = fig.legend(fontsize=20)
        return fig, axi, legend

class Tests:
    def __init__(self, name):
        self.name = name
        # data[results][intervals][variable][value]
        self.data = []
        self.data_columns = []
        self.data_units = []
        self.number_of_intervals = 0
        self.results = []
        self.plottable = []

    def SetPlottable(self, set_type, result_list, interval_list=None):
        if interval_list is None:
            for residx in result_list:
                for intidx in range(len(self.plottable[residx])):
                    for dataidx in range(len(self.plottable[residx][intidx])):
                        if self.plottable[residx][intidx][dataidx] != 'Deactivated':
                            self.plottable[residx][intidx][dataidx] = set_type
        else:
            for residx in result_list:
                for intidx in interval_list:
                    for dataidx in range(len(self.plottable[residx][intidx])):
                        if self.plottable[residx][intidx][dataidx] != 'Deactivated':
                            self.plottable[residx][intidx][dataidx] = set_type

    def FastPlot(self, varx, vary):
        varx = self.data_columns.index(varx)
        vary = self.data_columns.index(vary)

        fig = plt.figure(figsize=(19.2, 10.8))
        ax1 = fig.add_subplot(1,1,1)

        for residx in range(len(self.data)):
            for intidx in range(len(self.data[residx])):
                plotx = []
                ploty = []
                # Data to be plotted
                for elidx in range(len(self.plottable[residx][intidx])):
                    if self.plottable[residx][intidx][elidx] != 'Deactivated' and self.plottable[residx][intidx][elidx] != 'Off':
                        plotx.append(self.data[residx][intidx][varx][elidx])
                        ploty.append(self.data[residx][intidx][vary][elidx])
                if len(plotx) > 0:
                    ax1.plot(plotx, ploty, label=self.data_columns[vary], linewidth=3)

        if self.data_columns[vary] == 'Elastic Modulus':
            ax1.set_yscale('log')
            ax1.set_ylim(bottom=1)
        ax1.autoscale(True)
        ax1.set_ylabel(self.data_columns[vary] +' ' + self.data_units[vary], fontsize=20)
        ax1.set_xlabel(self.data_columns[varx] +' ' + self.data_units[varx], fontsize=20)
        ax1.set_title(self.name, fontsize=20)

        ax1.xaxis.set_tick_params(labelsize=20)
        ax1.yaxis.set_tick_params(labelsize=20)
        return fig, ax1

    def PlotAll(self):
        varx_timefig = self.data_columns.index('Time')
        varx_strainfig  = self.data_columns.index('Extensional Strain')
        unit_list_time = ['Force', 'Extensional Strain', 'Elastic Modulus', 'Extensional Strain Rate']
        unit_list_strain = ['Force', 'Elastic Modulus', 'Extensional Strain Rate']
        linewidth_time = [3, 3, 3, 0.3]
        linewidth_strain = [3, 3, 0.3]
                
        fig_time = plt.figure(figsize=(19.2, 10.8))
        fig_strain = plt.figure(figsize=(19.2, 10.8))
        # ----------
        # Time Part
        for i in range(len(unit_list_time)):
            vary = self.data_columns.index(unit_list_time[i])

            axi = fig_time.add_subplot(2,2,i+1)
            for residx in range(len(self.data)):
                for intidx in range(len(self.data[residx])):
                    plotx = []
                    ploty = []
                    # Data to be plotted
                    for elidx in range(len(self.plottable[residx][intidx])):
                        if self.plottable[residx][intidx][elidx] != 'Deactivated' and self.plottable[residx][intidx][elidx] != 'Off':
                            plotx.append(self.data[residx][intidx][varx_timefig][elidx])
                            ploty.append(self.data[residx][intidx][vary][elidx])
                    if len(plotx) > 0:
                        axi.plot(plotx, ploty, label=self.data_columns[vary], linewidth=linewidth_time[i])
                    if max(self.data[residx][intidx][vary]) < 1:
                        axi.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
                        axi.yaxis.get_offset_text().set_fontsize(20)

            if unit_list_time[i] == 'Elastic Modulus':
                axi.set_yscale('log')
                axi.set_ylim(bottom=1)
            axi.autoscale(True)
            axi.set_ylabel(self.data_columns[vary] +' ' + self.data_units[vary], fontsize=20)

            axi.xaxis.set_tick_params(labelsize=20)
            axi.yaxis.set_tick_params(labelsize=20)

        fig_time.suptitle(self.name + ' vs. Time', fontsize=40)
        fig_time.supxlabel(self.data_columns[varx_timefig] +' ' + self.data_units[varx_timefig], fontsize=20)
        # ----------
        # Strain Part
        for i in range(len(unit_list_strain)):
            vary = self.data_columns.index(unit_list_strain[i])

            axsi = fig_strain.add_subplot(2,2,i+1)
            for residx in range(len(self.data)):
                for intidx in range(len(self.data[0])):
                    plotx = []
                    ploty = []
                    # Data to be plotted
                    for elidx in range(len(self.plottable[residx][intidx])):
                        if self.plottable[residx][intidx][elidx] != 'Deactivated' and self.plottable[residx][intidx][elidx] != 'Off':
                            plotx.append(self.data[residx][intidx][varx_strainfig][elidx])
                            ploty.append(self.data[residx][intidx][vary][elidx])
                    if len(plotx) > 0:
                        axsi.plot(plotx, ploty, label=self.data_columns[vary], linewidth=linewidth_strain[i])
                    if max(self.data[residx][intidx][vary]) < 1:
                        axsi.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
                        axsi.yaxis.get_offset_text().set_fontsize(20)

            axsi.set_ylabel(self.data_columns[vary] +' ' + self.data_units[vary], fontsize=20)

            if unit_list_strain[i] == 'Elastic Modulus':
                axsi.set_yscale('log')
                axsi.set_ylim(bottom=1)
            axsi.autoscale(True)

            axsi.xaxis.set_tick_params(labelsize=20)
            axsi.yaxis.set_tick_params(labelsize=20)

        fig_strain.suptitle(self.name + ' vs. Strain', fontsize=40)
        fig_strain.supxlabel(self.data_columns[varx_strainfig] +' ' + self.data_units[varx_strainfig], fontsize=20)
        # ----------
        plt.tight_layout()

    def GetVal(self, var):
        varx = self.data_columns.index(var)
        return [[interval[varx] for interval in res] for res in self.data]
    
    def GetWholeTestData(self, var):
        varx = self.data_columns.index(var)
        outgoing_data = []
        for residx in range(len(self.data)):
            for intidx in range(len(self.data[residx])):
                outgoing_data += self.data[residx][intidx][varx]
        return outgoing_data
    
    # Yields the index for given value
    def GetValIndex(self, var, val, valtol=1, indextol=10):
        varx = self.data_columns.index(var)
        return_idx = []

        for residx in range(len(self.data)):
            for intidx in range(len(self.data[residx])):
                for idx in range(len(self.data[residx][intidx][varx])):
                    diff = abs(val - self.data[residx][intidx][varx][idx])
                    if diff < valtol:
                        return_idx.append([residx, intidx, idx])
                        # Remove multiple
                        for el in return_idx[:-1]:
                            if abs(el[-1] - idx) < indextol:
                                return_idx.pop(-1)
                                break
        return return_idx
    
    def GetValFor(self, var1, val1, var2, valtol=1, indextol=10):
        corr_idx = self.GetValIndex(var1, val1, valtol, indextol)[0]
        return self.GetVal(var2)[corr_idx[0]][corr_idx[1]][corr_idx[2]]
    
# General Functions
          
def MarkSpot(ax, test_no, x=None, y=None, color='red', horizontal=False, vertical=False,
             showNumbers=False):
    x_list = ax.lines[test_no].get_xdata()
    y_list = ax.lines[test_no].get_ydata()
    dist = 10000
    closest_idx = None
    if y is None:
        for elidx in range(len(x_list)):
            curr_dist = abs(x_list[elidx] - x)
            if curr_dist < dist:
                closest_idx = elidx
                dist = curr_dist
    elif x is None:
        for elidx in range(len(y_list)):
            curr_dist = abs(y_list[elidx] - y)
            if curr_dist < dist:
                closest_idx = elidx
                dist = curr_dist
    x = x_list[closest_idx]
    y = y_list[closest_idx]    
    ax.scatter(x, y, color=color, linewidth=3)
    if horizontal:
        ax.axhline(y=y, color=color, linestyle='solid', linewidth=1, alpha=0.5)
    if vertical:
        ax.axvline(x=x, color=color, linestyle='solid', linewidth=1, alpha=0.5)
    if showNumbers and horizontal:
        trans = transforms.blended_transform_factory(
            ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0, y, "{:.2e}".format(y), color=color, transform=trans, 
                ha="right", va="center", size=20)
    if showNumbers and vertical:
        trans = transforms.blended_transform_factory(
            ax.transData, ax.get_xticklabels()[0].get_transform())
        ax.text(x, 0, "{:.4f}".format(x), color=color, transform=trans, 
                ha="center", va="top", size=20)
    return ax, x, y

def ChangeAxUnits(ax, xory, constant):
    min_val = 10000
    max_val = 0
    if xory == 'x':
        for line in ax.lines:
            line.set_xdata([el*constant for el in line.get_xdata()])

            curr_line = line.get_xdata()
            curr_min = min(curr_line)
            curr_max = max(curr_line)

            if min_val > curr_min:
                min_val = curr_min
            if max_val < curr_max:
                max_val = curr_max

        ax.set_xlim(min_val*0.99, max_val*1.01)

    elif xory == 'y':
        for line in ax.lines:
            line.set_ydata([el*constant for el in line.get_ydata()])

            curr_line = line.get_ydata()
            curr_min = min(curr_line)
            curr_max = max(curr_line)

            if min_val > curr_min:
                min_val = curr_min
            if max_val < curr_max:
                max_val = curr_max

        ax.set_ylim(min_val*0.99, max_val*1.01)

    return ax

def OffsetData(ax, line_num, xory, val=None, first_el=None):
    if xory == 'y' and first_el:
        first_val = ax.lines[line_num].get_ydata()[0]
        ax.lines[line_num].set_ydata([el-first_val for el in ax.lines[line_num].get_ydata()])
        return ax, first_val
    
    elif xory == 'x' and first_el:
        first_val = ax.lines[line_num].get_xdata()[0]
        ax.lines[line_num].set_xdata([el-first_val for el in ax.lines[line_num].get_xdata()])
        return ax, first_val
    
    elif xory == 'x' and val:
        ax.lines[line_num].set_xdata([el+val for el in ax.lines[line_num].get_xdata()])
        return ax
    
    elif xory == 'y' and val:
        ax.lines[line_num].set_ydata([el+val for el in ax.lines[line_num].get_ydata()])
        return ax
    
    else:
        raise ValueError('Enter val or first_el as True')

# Send each portion separately!
# Procedure:
# 1) Max strain is the elastic max, difference between the next time step is the elastic portion
# 2) Starting from the next time step up until the asymptote of the viscous portion
# 2.a) Asymptote is calculated for %50-%100 of the viscous portion
# 3) Time to reach multiples of %10 of the viscous portion are recorded
# 3.a) [x,y,z,t,f] corresponds to strain for %10, %20, %30, %40 and %50.. of the time of recovery
#      11th element corresponds to entire recovery time
# AD-HOC PRONE in terms of intervals and results! Based on Test 21
def CalculateEVPortion(elastic_strain, viscous_strain, viscous_time):
    # # Elastic portion
    elastic_max = max(elastic_strain)
    elastic_min = viscous_strain[0]

    elastic_portion = elastic_max - elastic_min
    # # Viscous portion
    viscous_max = elastic_min
    viscous_max_idx = 0
    # Decision 2.a
    viscous_half_idx = int((len(viscous_strain)-viscous_max_idx)/2 + viscous_max_idx)
    # Calculating asymptote
    viscous_asymptote = 0
    el_cnt = 0
    for el in viscous_strain[viscous_half_idx:]:
        viscous_asymptote += el
        el_cnt += 1

    viscous_asymptote /= el_cnt
    viscous_asymptote_idx = viscous_strain.index(min(viscous_strain, key=lambda x:abs(x-viscous_asymptote)))

    viscous_portion = viscous_max - viscous_asymptote
    # # Relaxation Percentages
    # First time the asymptote is reached
    time_at_asymptote = viscous_time[viscous_asymptote_idx]
    time_at_viscous_beginning = viscous_time[0]
    time_to_recover = time_at_asymptote - time_at_viscous_beginning

    recover_list = []
    for i in range(0, 11):
        percent_time = min(viscous_time, key=lambda x:abs(x-(time_at_viscous_beginning + time_to_recover*(i/10))))
        recover_list.append(viscous_strain[viscous_time.index(percent_time)])

    returning_elastic = [elastic_max, elastic_min, elastic_portion]
    returning_viscous = [viscous_max, viscous_asymptote, viscous_portion]
    return returning_elastic, returning_viscous, recover_list, time_to_recover