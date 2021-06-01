from matplotlib import rc,rcParams,text
from matplotlib import pyplot as plt
rc('font', family='serif')
rc('font', size = 8)
rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('lines', linewidth=0.5)
rc('axes', linewidth=0.5)
rc('hatch', linewidth=0.15)
rc('xtick.major', width=0.5,size=2)
rc('xtick.minor', width=0.5,size=1)
rc('ytick.major', width=0.5,size=2)
rc('ytick.minor', width=0.5,size=1)
rc('xtick',top=True,bottom=True)
rc('ytick',left=True,right=True)
plt.rcParams["font.family"] = "serif"
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
rcParams.update(pgf_with_rc_fonts)

class LegendHandler(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendHandler, self).__init__()

    def legend_artist(self, legend, handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = text.Text(x0, y0, handle, **self.text_props,fontsize=fontsize)
        handlebox.add_artist(title)
        return title

class NellColors:
    def __init__(self):
            self.colors=['#2274A5','#FF9505','#397367','#D30C7B','#BC3908','#48BEFF','#553C3A','#7D54DE','#727472']
            self.black='#070600'
