"""
Coastal model module
"""

from copy import copy, deepcopy
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from striprtf.striprtf import rtf_to_text

from .utils import *

class OptimoorBase(object):
    """Base class for OPTIMOOR-related classes."""

    def __init__(self):
        
        # Initialisation
        self.src        = None
        self.name       = None
        self.fp         = None
        self.units      = None
        self._srcparse  = False

        return None

    @property
    def src(self) -> list[str]:
        return copy(self._src)
    
    @src.setter
    def src(self, value):
        self._src = value

    @property
    def name(self) -> str:
        return copy(self._name)
    
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def fp(self) -> str:
        return copy(self._fp)
    
    @fp.setter
    def fp(self, value):
        self._fp = value

    @property
    def units(self) -> str:
        return copy(self._units)
    
    @units.setter
    def units(self, value):
        self._units = value

class Vessel(OptimoorBase):
    """Class dedicated to performing vessel related operations for OPTIMOOR 
    files."""

    def __init__(self):
        
        # Initialisation
        super().__init__()
        #self.src        = None
        #self.name       = None
        #self.fp         = None
        #self.units      = None
        self.geometry   = None
        self.fairlead   = None
        #self._srcparse  = False

        return None
    
    def __str__(self):

        if self._srcparse:
            disp = "Vessel File for %s."%ANSI('Y', self.name)
        else:
            disp = "Empty Vessel File."
        return disp
    
    def __eq__(self, other):

        # Vessel classes are equal if the geometry and fairlead values are 
        # exactly equal
        if not isinstance(other, Vessel): return False
        else: return (self.geometry == other.geometry) \
                        & (self.fairlead == other.fairlead)
        
    def parse(self, src:list):
        """
        Parse and strip vessel geometric and parameter values from the RTF file.

        Parameters
        ----------
        src : list
            List of parsed rows for vessel data in the output RTF file
        """

        def try_convert(value):
            try: return float(value)
            except: return value

        # Renaming Dictionary
        rename_dict = {
            "LBP"                                       : "LBP",
            "Breadth"                                   : "Breadth",
            "Depth"                                     : "Depth",
            "Port Target"                               : "Target_Port",
            "Stbd Target"                               : "Target_Stbd",
            "End-on projected windage area"             : "Windage_End",
            "Side projected windage area"               : "Windage_Side",
            "Fendering possible from"                   : "Flatside_Fwd_Limit",
            "to"                                        : "Flatside_Aft_Limit",
            "Current drag data based on"                : "Drag_Current",
            "Wind drag data based on"                   : "Drag_Wind",
            "Wave motion data based on RAO data for"    : "Wave_Motion",
            "Roll Damping Coeff"                        : "Damping_Coef",
            "Loaded Cb"                                 : "Loaded_Cb"
        }

        # ======================================================================
        # DATA SOURCE, MANAGEMENT & PREPROCESSING
        # ======================================================================

        # Section Headers
        f               = src.copy()
        self.src        = src.copy()
        self.name       = f[0].split("for")[-1].strip()
        self.fp         = f[1].split("file")[-1].strip()[:-1]
        self.units      = f[2].split("Units in")[-1].strip().replace(" &", '')

        # Divider Locations
        wsidx       = np.argwhere([s == '' for s in f])[:,0]
        wsidx_aug   = np.append([-2], wsidx)
        wsidx_trim  = wsidx[(wsidx_aug[1:] - wsidx_aug[:-1]) != 1]

        usidx       = np.argwhere(["_______" in s for s in f])[:,0]
        
        # Partitioning the source file
        rows_geometry   = (wsidx_trim[0], wsidx_trim[1])
        rows_fairlead   = (usidx[0], usidx[1])


        # ======================================================================
        # VESSEL DIMENSIONS & GEOMETRY
        # ======================================================================
        
        # Initialisation
        geometry = {}
        geometry["Hullshape"] = {}

        # Geometric parameters
        for row in range(*rows_geometry):
            line = f[row]

            if ':' in line:
                key, val = [s.strip() for s in line.split(':')]
                val = ' '.join(val.split())
                geometry[rename_dict[key]] = try_convert(val)
                continue
            
            # Flatside Contours
            linesplit = line.split()
            if "X-dist" in linesplit:
                geometry["Hullshape"]['X'] = [float(x) for x in linesplit[1:]]
            elif "Depth" in linesplit:
                geometry["Hullshape"]['Z'] = [float(z) for z in linesplit[1:]]
            
        self.geometry = geometry


        # ======================================================================
        # FAIRLEAD DIMENSIONS & GEOMETRY
        # ======================================================================

        # Note: Assumes only one or no tails are used
        # Column names
        names = ["Line_No", "Fairlead_X", "Fairlead_Y", "Height_Above_Deck", 
            "Dist_to_Winch", "Brake_Limit", "Pre_Tension", "Line_Circumference",
            "Line_Type", "Line_Strength", "Tail_Length", "Tail_Circumference",
            "Tail_Type", "Tail_Strength"]
        count = [1, 1, 1, 1, 1, 1, 1, 3, 4]

        # Get index of last character in the header
        # Use this index to partition the "table"
        namerow     = f[rows_fairlead[0]+2]
        nameline    = namerow.split()
        nameline[1] += " %s"%nameline.pop(2)
        nameline[2] += " %s"%nameline.pop(3)
        nameidx     = [None] + [namerow.index(s) + len(s) for s in nameline]

        # Initialisation
        fairlead = []
        for row in range(rows_fairlead[0]+3, rows_fairlead[1]):
            line = f[row]

            # Extract text in table, then fill with N/A if data is missing
            strextract  = [line[s1:s2].split() for s1,s2 in 
                                        zip(nameidx[:-1], nameidx[1:])]
            strfillna   = [[np.nan for _ in range(c)] if len(l) != c else l 
                                    for l,c in zip(strextract, count)]
            strdissolve = [try_convert(s) for s in sum(strfillna, [])]
            fairlead.append(strdissolve)

        fairlead = list(zip(*fairlead))

        self.fairlead   = {key: val for key,val in zip(names, fairlead)}
        self._srcparse  = True
        
        return None

    @property
    def geometry(self) -> dict:
        return deepcopy(self._geometry)
    
    @geometry.setter
    def geometry(self, value):
        self._geometry = value

    @property
    def fairlead(self) -> dict:
        return copy(self._fairlead)
    
    @fairlead.setter
    def fairlead(self, value):
        self._fairlead = value

class Berth(OptimoorBase):
    """Class dedicated to performing berth related operations for OPTIMOOR 
    files."""

    def __init__(self):

        # Initialisation
        super().__init__()
        self.geometry   = None
        self.bollard    = None
        self.fender     = None

        return None
    
    def __str__(self):

        if self._srcparse:
            disp = "Berth File for %s."%ANSI('Y', self.name)
        else:
            disp = "Empty Berth File."
        return disp

    def __eq__(self, other):

        # Berth classes are equal if the geometry, bollard and fender values 
        # are exactly equal
        if not isinstance(other, Berth): return False
        else: return (self.geometry == other.geometry) \
                        & (self.bollard == other.bollard) \
                        & (self.fender == other.fender)
    
    def parse(self, src:list):
        """
        Parse and strip berth geometric and parameter values from the RTF file.

        Parameters
        ----------
        src : list
            List of parsed rows for berth data in the output RTF file
        """

        def try_convert(value):
            try: return float(value)
            except: return value

        # Renaming Dictionary
        rename_dict = {
            "Left to Right of Screen Site Plan Points"  : "Screen_Dir_L2R",
            "Width of Channel (for Current)"            : "Channel_Width",
            "Pier Height (Fixed) above Datum"           : "Pier_Elev",
            "Seabed Depth in way of Ship below Datum"   : "Seabed_Depth",
            "Permissible Surge Excursion Fwd/Aft"       : "Surge_Limit",
            "Permissible Sway Excursion Port/Stbd"      : "Sway_Limit",
            "Permissible Vertical Movement"             : "Heave_Limit",
            "Dist of Berth Target to Right of Origin"   : "Berth_Target_Dist",
            "Wind Speed Specified at Height"            : "WSPD_Height",
            "Current Specified at Depth"                : "Current_Depth"
        }


        # ======================================================================
        # DATA SOURCE, MANAGEMENT & PREPROCESSING
        # ======================================================================

        # Section Headers
        f               = src.copy()
        self.src        = src.copy()
        self.name       = f[0].split("for")[-1].strip()
        self.fp         = f[2].split("file")[-1].strip()[:-1]
        self.units      = f[3].split("Units in")[-1].strip().replace(" &", ',')

        # Divider Locations
        wsidx       = np.argwhere([s == '' for s in f])[:,0]
        wsidx_aug   = np.append([-2], wsidx)
        wsidx_trim  = wsidx[(wsidx_aug[1:] - wsidx_aug[:-1]) != 1]

        usidx       = np.argwhere(["_______" in s for s in f])[:,0]

        # Partitioning the source file
        rows_geometry   = (wsidx_trim[1], wsidx_trim[2])
        rows_bollard    = (usidx[0], usidx[1])
        rows_fender     = (usidx[2], usidx[3])
        

        # ======================================================================
        # BERTH DIMENSIONS & GEOMETRY
        # ======================================================================

        # Initialisation
        geometry = {}

        # Geometric parameters
        for row in range(*rows_geometry):
            line = f[row]

            if ':' in line:
                key, val = [s.strip() for s in line.split(':')]
                val = ' '.join(val.split())
                geometry[rename_dict[key]] = try_convert(val)
        
        self.geometry = geometry


        # ======================================================================
        # BOLLARD DIMENSIONS & GEOMETRY
        # ======================================================================

        # Column names
        bolnames = ["Hook/Bollard", "XDist_to_Origin", "Dist_to_Fender",
                    "Height_Above_Pier", "Allowable_Load"]
        
        # Get index of last character in the header
        # Use this index to partition the "table"
        namerow     = f[rows_bollard[0]+2]
        nameline    = namerow.split()
        nameline[1] += " %s"%nameline.pop(2)
        nameline[2] += " %s"%nameline.pop(3)
        nameidx     = [None] + [namerow.index(s) + len(s) for s in nameline]

        # Initialisation
        bollard = []
        for row in range(rows_bollard[0]+3, rows_bollard[1]):
            line = f[row] 

            # Extract text in table
            strextract = [line[s1:s2].strip() for s1,s2 in 
                                        zip(nameidx[:-1], nameidx[1:])]
            strconvert = [try_convert(s) for s in strextract]
            bollard.append(strconvert)

        bollard = list(zip(*bollard))

        self.bollard = {key: val for key,val in zip(bolnames, bollard)}


        # ======================================================================
        # FENDER DIMENSIONS & GEOMETRY
        # ======================================================================

        # Column names
        fendernames = ["Fender", "XDist_to_Origin", "Height_Above_Datum",
                       "Width_Along_Side", "Face_Contact_Area"]
        
        # Fender divider location
        fidx = np.argwhere(["Fender" in s for s 
                                in f[rows_fender[0]:rows_fender[1]]])[:,0]

        # Fender geometry
        # Get index of last character in the header to partition the table
        namerow     = f[rows_fender[0]+2]
        nameline    = namerow.split()
        nameline[0] += " %s"%nameline.pop(1)
        nameline[2] += " %s"%nameline.pop(3)
        nameline[3] += " %s"%nameline.pop(4)
        nameidx     = [None, 6] + [namerow.index(s) + len(s) for s in nameline]

        # Initialisation
        fender = []
        for row in range(rows_fender[0]+3, rows_fender[0]+fidx[1]-1):
            line = f[row]

            # Extract text in table
            strextract = [line[s1:s2].strip() for s1,s2 in 
                                        zip(nameidx[:-1], nameidx[1:])]
            strconvert = [try_convert(s) for s in strextract]
            fender.append(strconvert)
        
        fender = list(zip(*fender))
        fender = {key: val for key,val in zip(fendernames, fender)}

        # Fender compression data
        # Initialisation
        fcomp = {}
        for row in range(rows_fender[0]+fidx[1]+1, rows_fender[1], 3):
            line1, line2 = f[row], f[row+1]
            
            # Parse table
            str1 = [try_convert(s) for s in line1.split()][:-1]
            str2 = [try_convert(s) for s in line2.split()][:-1]
            fcomp["Fender_%s"%str1.pop(0)] = {"Force": str1, "Dist": str2}
        
        fender.update(fcomp)
        self.fender = fender

        self._srcparse = True

        return None
    
    @property
    def geometry(self) -> dict:
        return copy(self._geometry)
    
    @geometry.setter
    def geometry(self, value):
        self._geometry = value

    @property
    def bollard(self) -> dict:
        return copy(self._bollard)
    
    @bollard.setter
    def bollard(self, value):
        self._bollard = value
    
    @property
    def fender(self) -> dict:
        return deepcopy(self._fender)
    
    @fender.setter
    def fender(self, value):
        self._fender = value

class OPTSimulation(OptimoorBase):
    """Class with functionality to analyse OPTIMOOR simulation results."""

    def __init__(self):
        
        # Initialisation
        super().__init__()
        self.metocean   = None
        self.mline      = None
        self.fender     = None
        self.bollard    = None

        self.vessel     = None
        self.berth      = None

        return None
    
    def __str__(self):

        if self._srcparse:
            disp = "Simulation File for %s."%ANSI('Y', self.name)
        else:
            disp = "Empty Simulation File."
        return disp

    def __eq__(self, other):

        # Simulation files are equal if the underlying vessel, berth and input
        # metocean conditions are exactly equal
        if not isinstance(other, OPTSimulation): return False
        else: return (self.vessel == other.vessel) \
                        & (self.berth == other.berth) \
                        & (self.metocean == other.metocean)
        
    def add_berth(self, bth:Berth):
        """
        Attach a berth file to the simulation masterfile.

        Parameters
        ----------
        bth : Berth
            Berth layout of the OPTIMOOR simulation
        """

        if isinstance(bth, Berth): self.berth = bth
        return None

    def add_vessel(self, vsl:Vessel):
        """
        Attach a vessel file to the simulation masterfile

        Parameters
        ----------
        vsl : Vessel
            Vessel parameters of the OPTIMOOR simulation
        """

        if isinstance(vsl, Vessel): self.vessel = vsl
        return None
    
    def parse(self, src:list):
        """
        Parse and strip output from a single simulation from the RTF file.

        Parameters
        ----------
        src : list
            List of parsed rows for a single simulation in the output RTF file
        """

        def try_convert(value):
            try: return float(value)
            except: return value

        # Renaming Dictionary
        rename_dict = { 
            "Water Level"                   : "Water_Level",
            "Draft"                         : "Draft",
            "Trim"                          : "Trim",
            "GM"                            : "GM",
            "Bottom Clearance"              : "Bottom_Clearance",
            "Fwd Offset of Vessel Target"   : "VSL_Fwd_Offset",
            "Vessel Port Target"            : "VSL_Vrt_Offset",
            "Significant Wave Ht"           : "Hs",
            "Wave Mean Period"              : "Tm",
            "Wave Direction from"           : "MWD",
            "Wave Spectrum"                 : "Wave_Spectrum",
            "Current"                       : "CSPD",
            "Current Direction from"        : "CDIR",
            "Wind Speed"                    : "WSPD",
            "Wind Direction from"           : "WDIR",
            "Total End-on Windage Area"     : "Windage_End",
            "Total Side Windage Area"       : "Windage_Side"
        }


        # ======================================================================
        # DATA SOURCE, MANAGEMENT & PREPROCESSING
        # ======================================================================

        # Section Headers
        f               = src.copy()
        self.src        = src.copy()
        self.fp         = f[2].split("file")[-1].strip()[:-1]
        self.units      = f[3].split("Units in")[-1].strip().replace(" &", ',')
        self.name       = f[5].split("Ref:")[-1].strip()

        # Divider Locations
        wsidx       = np.argwhere([s == '' for s in f])[:,0]
        wsidx_aug   = np.append([-2], wsidx)
        wsidx_trim  = wsidx[(wsidx_aug[1:] - wsidx_aug[:-1]) != 1]

        usidx       = np.argwhere(["_______" in s for s in f])[:,0]

        # Partitioning the source file
        rows_metocean   = (wsidx[1], wsidx[3])
        rows_forcing    = (wsidx[3], wsidx[4])
        rows_movement   = (wsidx[4], wsidx[5])
        rows_mline      = (usidx[0], usidx[1])
        rows_fender     = (usidx[1], usidx[2])
        rows_bollard    = (usidx[2], usidx[3])


        # ======================================================================
        # METOCEAN DATA INPUT & SETUP
        # ======================================================================

        # Initialisation
        metocean = {}

        # MetOcean Parameters
        for row in range(*rows_metocean):
            line = f[row]
            
            if ':' in line:
                key, val = [s.strip() for s in line.split(':')]
                val = ' '.join(val.split())
                metocean[rename_dict[key]] = try_convert(val)

        # MetOcean Forcing
        namerow = f[rows_forcing[0]+1]
        nameline = namerow.split()
        nameline[2] += " %s"%nameline.pop(3)
        nameidx = [namerow.index(nameline[0])] \
                        + [namerow.index(s) + len(s) for s in nameline]

        # Initialisation
        headers = []; forcing = []
        for row in range(rows_forcing[0]+2, rows_forcing[1]):
            line = f[row]

            # Row Headers
            headers.append(line.split(':')[0].strip())

            # Extract text in table
            strextract = [line[s1:s2].strip() for s1,s2 in 
                                    zip(nameidx[:-1], nameidx[1:])]
            strconvert = [try_convert(s) for s in strextract]
            forcing.append(strconvert)
        
        forcing = list(zip(*forcing))
        col = [{key: val for key,val in zip(headers, fi)} for fi in forcing]
        allforcing = {key: val for key,val in zip(nameline, col)}

        metocean["Forcing"] = allforcing

        self.metocean = metocean


        # ======================================================================
        # MOORING LINE LOADS
        # ======================================================================

        # Column names
        mlnames = ["Fairlead", "Bollard", "Pull-in", "Length", "In-Line_Motion",
                "Winch_Slip", "Downward_Inclination", "Tension", "Strength_Pct"]
        
        # Get index of last character in the header to partition the table
        namerow = f[rows_mline[0]+1]
        nameline = namerow.split()
        nameline[0] += " %s"%nameline.pop(1)
        mlidx = [ii+4 for ii in range(len(namerow)-4) if namerow[ii:ii+4] == "Line"]
        nameidx = [None] + [namerow.index(s) + len(s) if "Line" not in s
                                else mlidx.pop(0) for s in nameline]
        nameidx[1] += 3
        
        # Initialisation
        mline = []
        for row in range(rows_mline[0]+3, rows_mline[1]):
            line = f[row]
            
            # Extract text in table
            strextract = [line[s1:s2].strip() for s1,s2 in 
                                        zip(nameidx[:-1], nameidx[1:])]
            strfillna = [np.nan if len(s) == 0 else try_convert(s) 
                                        for s in strextract]
            strconvert = strfillna.pop(0).split('-') + strfillna
            mline.append(strconvert)

        mline = list(zip(*mline))
        self.mline = {key: val for key,val in zip(mlnames, mline)}
        

        # ======================================================================
        # VESSEL MOVEMENTS
        # ======================================================================
        
        dimensions = ["Surge", "Sway", "Roll", "Heave"]

        motion = []
        for row in range(rows_movement[0]+1, rows_movement[1]):
            line = f[row]
            
            # Divide table
            if len(line.split(')')) == 1: 
                motion.append([np.nan, np.nan, np.nan, np.nan])
            else:
                motion.append([s.split('(')[0].split()[-1].replace("\u00b0", '') 
                                    for s in line.split(')')[:-1]])
        motion = [(try_convert(s1), try_convert(s2)) for s1,s2 in 
                                    zip(motion[1], motion[0])]
        
        self.motion = {key: val for key,val in zip(dimensions, motion)}
        

        # ======================================================================
        # FENDER LOADS
        # ======================================================================

        # Column names
        # Also get index of last character in the header to partition the table
        namerow     = f[rows_fender[0]+1]
        nameline    = namerow.split()
        nameline[4] += " %s"%nameline.pop(5)
        nameidx     = [None] + [namerow.index(s) + len(s) for s in nameline]
        
        # Initialisation
        fender = []
        for row in range(rows_fender[0]+2, rows_fender[1]-1):
            line = f[row]

            # Extract text in table
            strextract = [line[s1:s2].strip() for s1,s2 in 
                                        zip(nameidx[:-1], nameidx[1:])]
            strconvert = [np.nan if len(s) == 0 else try_convert(s) 
                                        for s in strextract]
            fender.append(strconvert)

        fender = list(zip(*fender))

        self.fender = {key: val for key,val in zip(nameline,fender)}


        # ======================================================================
        # BOLLARD LOADS
        # ======================================================================

        # Column names
        bldnames = ["Hook/Bollard", "Fx", "Fy", "Fx_Other", "Fy_Other", "Total",
                    "Bollard_Strength_Pct", "Plan_Dir", "Uplift"]
        
        # Get index of last character in the header to partition the table
        # Beware some columns have repeat headings
        namerow     = f[rows_bollard[0]+2]
        nameline    = namerow.split()
        nameline[7] += " %s"%nameline.pop(8)
        forceidx    = [ii+5 for ii in range(len(namerow)-5) 
                                            if namerow[ii:ii+5] == "Force"]
        nameidx     = [None] + [namerow.index(s) + len(s) if s != "Force" 
                                else forceidx.pop(0) for s in nameline]

        # Initialisation
        bollard = []
        for row in range(rows_bollard[0]+3, rows_bollard[1]):
            line = f[row]
            
            # Extract text in table, then fill N/A if data is missing
            strextract = [line[s1:s2].strip() for s1,s2 in 
                                        zip(nameidx[:-1], nameidx[1:])]
            strconvert = [np.nan if len(s) == 0 else try_convert(s) 
                                        for s in strextract]
            bollard.append(strconvert)
        
        bollard = list(zip(*bollard))
        
        self.bollard = {key: val for key,val in zip(bldnames, bollard)}

        self._srcparse = True

        return None

    def plot_geometry(self) -> Figure:
        """
        Plots the OPTIMOOR setup in plan, side and end view. Coordinates are 
        vessel-based.

        Returns
        -------
        fig : Figure
            Figure showing the OPTIMOOR setup and connections between elements
        """

        # Set hatch linewidth
        mpl.rcParams["hatch.linewidth"] = 1.5


        # ======================================================================
        # DATA PROCESSING 
        # ======================================================================

        # Pull geometry from constituent files
        vsl = self.vessel
        bth = self.berth
        met = self.metocean
        ml  = self.mline

        # Basic Geometry & Water Levels
        L = vsl.geometry["LBP"]
        B = vsl.geometry["Breadth"]

        wsel = float(met["Water_Level"].split("above")[0])
        draft = float(met["Draft"].split('(')[0])
        depth = vsl.geometry["Depth"]
        deck_elev = wsel + depth - draft
        keel_elev = wsel - draft

        HS = np.c_[vsl.geometry["Hullshape"]['X'], vsl.geometry["Hullshape"]['Z']]
        HS = np.vstack((HS[0,:], HS, HS[-1,:]))
        HS[0,1] = 0; HS[-1,1] = 0
        HS[:,1] = deck_elev - HS[:,1]

        Vkwargs = {"facecolor": "#fff0f0", "edgecolor": 'k', "label": "Vessel"}
        Vplan = Rectangle((-L/2, -B/2), L, B, **Vkwargs)
        Vside = Rectangle((-L/2, keel_elev), L, depth, **Vkwargs)
        Vfs   = Polygon(HS, facecolor="none", edgecolor='k', linestyle="--",
                        hatch="/", linewidth=1.5, label="Flatside")
        Vend  = Rectangle((-B/2, wsel-draft), B, depth, **Vkwargs)

        # Fairlead Geometry
        fairlead = pd.DataFrame(vsl.fairlead)
        FLx = fairlead["Fairlead_X"].values
        FLy = fairlead["Fairlead_Y"].values
        FLz = deck_elev + fairlead["Height_Above_Deck"].values
        
        # Bollard Geometry
        # Sometimes the batch result file will not have pier elevation data.
        # In this case the user will have to manually specify the value.
        try:
            pier_elev = bth.geometry["Pier_Elev"]; 
        
        except KeyError:
            errormsg = "Pier elevation above datum missing. Please enter a " \
                + "value manually via the .add_pier_elev() method.\n"
            warnings.warn(ANSI('Y', errormsg))

            return None

        bollard = pd.DataFrame(bth.bollard)
        BLx = bollard["XDist_to_Origin"].values
        BLy = B/2 + bollard["Dist_to_Fender"].values
        BLz = pier_elev + bollard["Height_Above_Pier"].values
        
        # Fender Geometry
        FDx = bth.fender["XDist_to_Origin"]
        FDy = B/2 + np.zeros_like(FDx)
        FDz = bth.fender["Height_Above_Datum"]

        # Mooring Lines
        mooring = pd.DataFrame(ml)
        MLx = []
        MLy = []
        MLz = []
        for _, row in mooring.iterrows():
            flidx = np.argwhere(fairlead["Line_No"] == float(row["Fairlead"]))[0,0]
            blidx = np.argwhere(bollard["Hook/Bollard"] == row["Bollard"])[0,0]
            
            MLx.append([FLx[flidx], BLx[blidx]])
            MLy.append([FLy[flidx], BLy[blidx]])
            MLz.append([FLz[flidx], BLz[blidx]])


        # ======================================================================
        # FIGURE PLOTTING
        # ======================================================================

        # Initialise figure
        fig = plt.figure(figsize=(13.33, 7.5))
        gs = fig.add_gridspec(2,2)
        ax0 = fig.add_subplot(gs[1,0])
        ax1 = fig.add_subplot(gs[0,0], sharex=ax0)
        ax2 = fig.add_subplot(gs[:,1])
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Plan View
        ax0.add_patch(Vplan)
        [ax0.plot(MLxi, MLyi, c='k', label="__nolegend_") 
                                    for MLxi,MLyi in zip(MLx, MLy)]
        ax0.scatter(FLx, FLy, facecolor='r', edgecolor='k')
        ax0.scatter(BLx, BLy, facecolor='k', edgecolor='k')
        ax0.scatter(FDx, FDy, facecolor='g', edgecolor='k')
        ax0.set(xlabel='X', ylabel='Y', title="Plan View")
        ax0.axis("equal")

        # Side View
        ax1.add_patch(Vside)
        ax1.add_patch(Vfs)
        [ax1.plot(MLxi, MLzi, c='k', label="__nolegend_") 
                                    for MLxi,MLzi in zip(MLx, MLz)]
        ax1.scatter(FLx, FLz, facecolor='r', edgecolor='k')
        ax1.scatter(BLx, BLz, facecolor='k', edgecolor='k')
        ax1.scatter(FDx, FDz, facecolor='g', edgecolor='k')
        ax1.axhline(wsel, c="C0")
        ax1.set(ylabel="Z Above Datum", title="Side View")
        ax1.axis("equal")

        # End View
        ax2.add_patch(Vend)
        [ax2.plot(MLyi, MLzi, c='k', label="__nolegend_") 
                                    for MLyi,MLzi in zip(MLy, MLz)]
        ax2.scatter(FLy, FLz, facecolor='r', edgecolor='k', label="Fairlead")
        ax2.scatter(BLy, BLz, facecolor='k', edgecolor='k', label="Bollard")
        ax2.scatter(FDy, FDz, facecolor='g', edgecolor='k', label="Fender")
        ax2.axhline(wsel, c="C0")
        ax2.set(xlabel='Y', ylabel="Z Above Datum", title="End View")
        handles, labels = ax2.get_legend_handles_labels()
        handles.append(Vfs)
        labels.append(Vfs.get_label())
        ax2.legend(handles, labels)
        ax2.axis("equal")

        plt.tight_layout()
        
        return fig
    
    @property
    def metocean(self) -> dict:
        return deepcopy(self._metocean)
    
    @metocean.setter
    def metocean(self, value):
        self._metocean = value
    
    @property
    def motion(self) -> dict:
        return copy(self._motion)
    
    @motion.setter
    def motion(self, value):
        self._motion = value
    
    @property
    def mline(self) -> dict:
        return copy(self._mline)
    
    @mline.setter
    def mline(self, value):
        self._mline = value

    @property
    def fender(self) -> dict:
        return copy(self._fender)
    
    @fender.setter
    def fender(self, value):
        self._fender = value

    @property
    def bollard(self) -> dict:
        return copy(self._bollard)
    
    @bollard.setter
    def bollard(self, value):
        self._bollard = value
    
    @property
    def vessel(self) -> Vessel:
        return deepcopy(self._vessel)
    
    @vessel.setter
    def vessel(self, value):
        self._vessel = value

    @property
    def berth(self) -> Berth:
        return deepcopy(self._berth)
    
    @berth.setter
    def berth(self, value):
        self._berth = value

class Optimoor(object):
    """
    Top level container for all OPTIMOOR-related functionality. 
    
    This class provides simple and user-friendly commands with the intent for
    users to streamline the mooring analysis workflow and dealing with batch
    operation output RTF files.

    Note: This piece of code is still under active development, and may not work 
    for all Optimoor batch run results. 
    """

    def __init__(self):
        
        # Format warning text
        warnings.formatwarning = warning_output

        # Initialisation
        self.src        = None
        self.vessel     = None
        self.berth      = None
        self.batch      = None
        self._srcparse = False

        return None
    
    def __str__(self):

        T = "\u251c\u2500\u2500"
        L = "\u2514\u2500\u2500"

        if self._srcparse:
            disp = "Collection of batch run files for %s at %s.\n" \
                %(ANSI('Y', self.vessel.name), ANSI('Y', self.berth.name)) \
                + "Users beware: this piece of code is still under " \
                + "active development.\n\n" \
                + "File Structure: " \
                + "(Key: %s %s %s)\n"%(ANSI('G', "Class"), ANSI('Y', "Method"), 
                                                ANSI('R', "List")) \
                + "%s\n"%ANSI('G', "Optimoor") \
                + "%s %s\n"%(T, ANSI('G', "vessel")) \
                + "\u2502   %s src, name, fp, units, geometry, fairlead\n"%L \
                + "%s %s\n"%(T, ANSI('G', "berth")) \
                + "\u2502   %s src, name, fp, units, geometry, bollard, fender\n"%L \
                + "%s %s\n"%(T, ANSI('R', "batch")) \
                + "\u2502   %s %s\n"%(T, ANSI('G', "Simulation 1")) \
                + "\u2502   \u2502   %s src, name, fp, units, metocean, "%T \
                + "motion, mline, fender, bollard, vessel, berth\n" \
                + "\u2502   \u2502   %s %s\n"%(L, ANSI('Y', "plot_geometry()")) \
                + "\u2502   %s %s\n"%(T, ANSI('G', "...")) \
                + "\u2502   %s %s\n"%(L, ANSI('G', "Simulation n")) \
                + "%s %s\n"%(T, ANSI('Y', "add_pier_elev()")) \
                + "%s %s"%(L, ANSI('Y', "convertref()"))
        else:
            disp = "Empty Ensemble File."

        return disp

    def __eq__(self, other):

        # Optimoor files are equal if the underlying vessel, berth and batch
        # simulation files are exactly the same
        if not isinstance(other, Optimoor): return False
        else: return (self.vessel == other.vessel) \
                        & (self.berth == other.berth) \
                        & (self.batch == other.batch)

    def _metocean_summary(self):
        """Computes a summary of metocean inputs for all batch runs."""

        # Ideal columns - some simulations might have missing columns
        idealcol = ["WSPD", "WDIR", "CSPD", "CDIR", "Hs", "Tm", "MWD", 
                    "Water_Level", "Draft", "Trim"]

        # Pull and concatenate input metocean parameters from data file
        metsum = []
        for sim in self.batch:
            met = sim.metocean
            del met["Forcing"]
            metsum.append(pd.DataFrame(data=met.values(), index=met.keys()).T)
        metsum = pd.concat(metsum, ignore_index=True)

        # Subset and clean data
        colnames = [c for c in idealcol if c in metsum.columns]
        metsum = metsum[colnames]
        for col in colnames:
            try: 
                metsum[col] = metsum[col].str.split().str[0].str.split("\u00b0")\
                                    .str[0].astype(float)
            except AttributeError:
                metsum[col] = metsum[col].astype(float)

        self.metsummary = metsum

        return None
    
    def add_pier_elev(self, elevation:float):
        """
        Adds the pier elevation to the ensemble if it is not included in the 
        simulation output file.

        Parameters
        ----------
        elevation : float
            Pier elevation above datum in the same units as the model
        """

        if not self._srcparse:
            print("Empty berth file.")
            return None
        
        # Obtain berth data and update value
        B           = self.berth
        geometry    = B.geometry
        geometry.update({"Pier_Elev": elevation})

        # Rewrite berth data in berth and simulation files
        B.geometry  = geometry
        self.berth  = B
        batch       = self.batch
        [S.add_berth(B) for S in batch]
        self.batch  = batch

        return None

    @staticmethod
    def convertrtf(src) -> list[str]:
        """
        Converts RTF text to a list of strings.

        Parameters
        ----------
        src : opened file
            Opened RTF file as obtained through the "open()" command

        Returns
        -------
        ln : list
            List of converted RTF strings
        """
        
        # Read and convert
        f = src.read()
        fstr = rtf_to_text(f)
        ln = fstr.split("\n")

        return ln
    
    def parse(self, src:list):
        """
        Parse and strip outputs from a batch simulation RTF file. The 
        convertrtf() function should first be used to convert the output file
        to the correct format.

        Parameters
        ----------
        src : list
            List of RTF-converted rows for a batch simulation containing vessel,
            berth, and simulation outputs
        """

        # ======================================================================
        # DATA SOURCE, MANAGEMENT & PREPROCESSING
        # ======================================================================

        # Section Headers
        f           = src.copy()
        self.src    = src.copy()

        # Warning messages
        vslwarn = "Multiple vessel files detected. " \
                        + "The first parsed file will be used.\n"
        bthwarn = "Multiple berth files detected. " \
                        + "The first berth file will be used.\n"

        # Divider Locations
        vslidx  = np.argwhere(["Vessel Data for" in s for s in f])[:,0]
        bthidx  = np.argwhere(["Berth Data for" in s for s in f])[:,0]
        simidx  = np.argwhere(["Static Mooring Response" in s for s in f])[:,0]
        simidx  = simidx.tolist() + [None]
        
        if len(vslidx) > 1: warnings.warn(ANSI('Y', vslwarn), UserWarning)
        if len(bthidx) > 1: warnings.warn(ANSI('Y', bthwarn), UserWarning)
        
        # Partitioning the source file
        src_vsl = f[vslidx[0]:bthidx[0]]
        src_bth = f[bthidx[0]:simidx[0]]
        src_sim = [f[s1:s2] for s1,s2 in zip(simidx[:-1], simidx[1:])]
        
        # Parse data
        V = Vessel()
        V.parse(src_vsl)

        B = Berth()
        B.parse(src_bth)

        batch = []
        for src in src_sim:
            S = OPTSimulation()
            S.parse(src)
            S.add_vessel(V)
            S.add_berth(B)
            batch.append(S)

        self.vessel = V
        self.berth  = B
        self.batch  = batch

        # Generate summary table for metocean conditions
        self._metocean_summary()

        self._srcparse = True

        return None
    
    @property
    def vessel(self) -> Vessel:
        return deepcopy(self._vessel)
    
    @vessel.setter
    def vessel(self, value):
        self._vessel = value

    @property
    def berth(self) -> Berth:
        return deepcopy(self._berth)
    
    @berth.setter
    def berth(self, value):
        self._berth = value

    @property
    def batch(self) -> list[OPTSimulation]:
        return deepcopy(self._batch)
    
    @batch.setter
    def batch(self, value):
        self._batch = value

    @property
    def metsummary(self) -> pd.DataFrame:
        return copy(self._metsum)
    
    @metsummary.setter
    def metsummary(self, value):
        self._metsum = value


