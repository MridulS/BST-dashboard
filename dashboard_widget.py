import ipywidgets as widgets
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
import matplotlib.pyplot as plt
import numpy as np

# Import default parameter values (init_idiosyncratic_shock)
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks as base_params

# Set the parameters for the baseline results in the paper
# using the variable values defined in the cell above
base_params['PermGroFac'] =                [1.03] # Permanent income growth factor
base_params['Rfree']      = Rfree        =  1.04  # Interest factor on assets
base_params['DiscFac']    = DiscFac      =  0.96  # Time Preference Factor
base_params['CRRA']       = CRRA         =  2.00  # Coefficient of relative risk aversion
base_params['UnempPrb']   = UnempPrb     =  0.005 # Probability of unemployment (e.g. Probability of Zero Income in the paper)
base_params['IncUnemp']   = IncUnemp     =  0.0   # Induces natural borrowing constraint
base_params['PermShkStd'] =                 [0.1]   # Standard deviation of log permanent income shocks
base_params['TranShkStd'] =                 [0.1]   # Standard deviation of log transitory income shocks

# Uninteresting housekeeping and details
# Make global variables for the things that were lists above -- uninteresting housekeeping
PermGroFac, PermShkStd, TranShkStd = base_params['PermGroFac'][0],base_params['PermShkStd'][0],base_params['TranShkStd'][0]

# Some technical settings that are not interesting for our purposes
base_params['LivPrb']       = [1.0]   # 100 percent probability of living to next period
base_params['CubicBool']    = True    # Use cubic spline interpolation
base_params['T_cycle']      = 1       # No 'seasonal' cycles
base_params['BoroCnstArt']  = None    # No artificial borrowing constraint




# Define a slider for the discount factor
DiscFac_widget = widgets.FloatSlider(
    min=0.9,
    max=0.99,
    step=0.0002,
    value=DiscFac, # Default value
    continuous_update=False,
    readout_format='.4f',
    description='\u03B2') # beta unicode

# Define a slider for relative risk aversion
CRRA_widget = widgets.FloatSlider(
    min=1.0,
    max=5.0,
    step=0.01,
    value=CRRA,  # Default value
    continuous_update=False,
    readout_format='.2f',
    description='\u03C1') #rho unicode

# Define a slider for the interest factor
Rfree_widget = widgets.FloatSlider(
    min=1.01,
    max=1.08,
    step=0.001,
    value=Rfree,  # Default value
    continuous_update=False,
    readout_format='.4f',
    description='R')


# Define a slider for permanent income growth
PermGroFac_widget = widgets.FloatSlider(
    min=1.00,
    max=1.08,
    step=0.001,
    value=PermGroFac,  # Default value
    continuous_update=False,
    readout_format='.4f',
    description='\u0393') # capital gamma

# Define a slider for unemployment (or retirement) probability
UnempPrb_widget = widgets.FloatSlider(
    min=0.0001,
    max=0.01, # Go up to twice the default value
    step=0.00001,
    value=UnempPrb,
    continuous_update=False,
    readout_format='.5f',
    description='℘')
    
# Define a slider for unemployment (or retirement) probability
IncUnemp_widget = widgets.FloatSlider(
    min=0.0001,
    max=0.01, # Go up to twice the default value
    step=0.00001,
    value=IncUnemp,
    continuous_update=False,
    readout_format='.5f',
    description='$\\mho$')


def makeConvergencePlot(DiscFac,CRRA,Rfree,PermGroFac,UnempPrb):
    baseEx = IndShockConsumerType(**base_params)
    # model=baseEx
    mMax=11
    mMin=0
    cMin=0
    cMax=7
    baseEx.DiscFac = DiscFac
    baseEx.CRRA = CRRA
    baseEx.Rfree = Rfree
    baseEx.PermGroFac = [PermGroFac]
    baseEx.UnempPrb = UnempPrb
    baseEx.cycles = 100
#     print(DiscFac, CRRA, Rfree, PermGroFac, UnempPrb, baseEx.cycles)

    baseEx.solve(verbose=False)    
    baseEx.unpackcFunc()
    plt.figure(figsize = (12,8))
    plt.ylim([cMin,cMax])
    plt.xlim([mMin,mMax])
    
    m1    = np.linspace(0, 9.5, 1000) # Set the plot range of m
    m2    = np.linspace(0, 6.5, 500)
    c_m   = baseEx.cFunc[0](m1)   # c_m can be used to define the limiting inﬁnite-horizon consumption rule here
    c_t1  = baseEx.cFunc[-2](m1) # c_t1 defines the second-to-last period consumption rule
    c_t5  = baseEx.cFunc[-6](m1) # c_t5 defines the T-5 period consumption rule
    c_t10 = baseEx.cFunc[-11](m1)  # c_t10 defines the T-10 period consumption rule
    c_t0  = m2                            # c_t0 defines the last period consumption rule
    plt.plot(m1, c_m, label='$c(m)$')
    plt.plot(m1, c_t1, label='$c_{T-1}(m)$')
    plt.plot(m1, c_t5, label='$c_{T-5}(m)$')
    plt.plot(m1, c_t10, label='$c_{T-10}(m)$')
    plt.plot(m2, c_t0, label='$c_{T}(m) = 45$ degree line')
    plt.legend()
    plt.tick_params(labelbottom=False, labelleft=False,left='off',right='off',bottom='off',top='off')

    plt.show()
    return None



def makeGICFailExample(Rfree, PermGroFac):
    GIC_fail_dictionary = dict(base_params)
    GIC_fail_dictionary['Rfree']      = Rfree
    GIC_fail_dictionary['PermGroFac'] = [PermGroFac]

    GICFailExample = IndShockConsumerType(
        cycles=0, # cycles=0 makes this an infinite horizon consumer
        **GIC_fail_dictionary)

    # Calculate "Sustainable" consumption that leaves expected m unchanged
    # In the perfect foresight case, this is just permanent income plus interest income
    # A small adjustment is required to take account of the consequences of uncertainty
    # See "Growth Patience and the GIC" above
    InvEpShkInvAct = np.dot(GICFailExample.PermShkDstn[0].pmf, GICFailExample.PermShkDstn[0].X**(-1))
    InvInvEpShkInvAct = (InvEpShkInvAct) ** (-1)                      # $E[\psi^{-1}]$
    PermGroFacAct = GICFailExample.PermGroFac[0] * InvInvEpShkInvAct # $(E[\psi^{-1}])^{-1}$
    ERnrm   = GICFailExample.Rfree / PermGroFacAct # Interest factor normalized by uncertainty-adjusted growth
    Ernrm   = ERnrm - 1                            # Interest rate is interest factor - 1
    mSSfunc = lambda m : 1 + (m-1)*(Ernrm/ERnrm)   # "sustainable" consumption: consume your (discounted) interest income

    GICFailExample.solve() # Above, we set up the problem but did not solve it
    GICFailExample.unpackcFunc()  # Make the consumption function easily accessible for plotting
    m = np.linspace(0,5,1000)
    c_m = GICFailExample.cFunc[0](m)
    E_m = mSSfunc(m)
    plt.figure(figsize = (12,8))
    plt.plot(m,c_m, label="$c(m_{t})$")
    plt.plot(m,E_m, label='$\mathsf{E}_{t}[\Delta m_{t+1}] = 0$')
    plt.legend()
    plt.xlim(0,5.5)
    plt.ylim(0,1.6)
    plt.tick_params(labelbottom=False, labelleft=False,left='off',right='off',bottom='off',top='off')
    plt.show()
    return None


# Define a function to construct the arrows on the consumption growth rate function
def arrowplot(axes, x, y, narrs=15, dspace=0.5, direc='neg',
              hl=0.01, hw=3, c='black'):
    '''
    The function is used to plot arrows given the data x and y.

    Input:
        narrs  :  Number of arrows that will be drawn along the curve

        dspace :  Shift the position of the arrows along the curve.
                  Should be between 0. and 1.

        direc  :  can be 'pos' or 'neg' to select direction of the arrows

        hl     :  length of the arrow head

        hw     :  width of the arrow head

        c      :  color of the edge and face of the arrow head
    '''

    # r is the distance spanned between pairs of points
    r = np.sqrt(np.diff(x)**2+np.diff(y)**2)
    r = np.insert(r, 0, 0.0)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = np.cumsum(r)

    # based on narrs set the arrow spacing
    aspace = r.sum() / narrs

    if direc is 'neg':
        dspace = -1.*abs(dspace)
    else:
        dspace = abs(dspace)

    arrowData = [] # will hold tuples of x,y,theta for each arrow
    arrowPos = aspace*(dspace) # current point on walk along data
                                 # could set arrowPos to 0 if you want
                                 # an arrow at the beginning of the curve

    ndrawn = 0
    rcount = 1
    while arrowPos < r.sum() and ndrawn < narrs:
        x1,x2 = x[rcount-1],x[rcount]
        y1,y2 = y[rcount-1],y[rcount]
        da = arrowPos-rtot[rcount]
        theta = np.arctan2((x2-x1),(y2-y1))
        ax = np.sin(theta)*da+x1
        ay = np.cos(theta)*da+y1
        arrowData.append((ax,ay,theta))
        ndrawn += 1
        arrowPos+=aspace
        while arrowPos > rtot[rcount+1]:
            rcount+=1
            if arrowPos > rtot[-1]:
                break

    for ax,ay,theta in arrowData:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta)*hl/2.0 + ax
        dy0 = np.cos(theta)*hl/2.0 + ay
        dx1 = -1.*np.sin(theta)*hl/2.0 + ax
        dy1 = -1.*np.cos(theta)*hl/2.0 + ay

        if direc is 'neg' :
            ax0 = dx0
            ay0 = dy0
            ax1 = dx1
            ay1 = dy1
        else:
            ax0 = dx1
            ay0 = dy1
            ax1 = dx0
            ay1 = dy0

        axes.annotate('', xy=(ax0, ay0), xycoords='data',
                xytext=(ax1, ay1), textcoords='data',
                arrowprops=dict( headwidth=hw, frac=1., ec=c, fc=c))
        

def makesomethingplot(Rfree, PermGroFac, DiscFac, CRRA):
    # cycles=0 tells the solver to find the infinite horizon solution

    baseEx_inf = IndShockConsumerType(cycles=0,**base_params)
    baseEx_inf.Rfree = Rfree
    baseEx_inf.PermGroFac = [PermGroFac]
    baseEx_inf.CRRA = CRRA
    baseEx_inf.DiscFac = DiscFac 
    baseEx_inf.solve()
    baseEx_inf.unpackcFunc()
    # Define a function to calculate expected consumption
    def exp_consumption(a):
        '''
        Taking end-of-period assets a as input, return expectation of next period's consumption
        Inputs:
           a: end-of-period assets
        Returns:
           expconsump: next period's expected consumption
        '''
        GrowFac_tp1 = baseEx_inf.PermGroFac[0]* baseEx_inf.PermShkDstn[0].X
        Rnrm_tp1 = baseEx_inf.Rfree / GrowFac_tp1
        # end-of-period assets plus normalized returns
        b_tp1 = Rnrm_tp1*a
        # expand dims of b_tp1 and use broadcasted sum of a column and a row vector
        # to obtain a matrix of possible beginning-of-period assets next period
        # This is much much faster than looping
        m_tp1 = np.expand_dims(b_tp1, axis=1) + baseEx_inf.TranShkDstn[0].X
        part_expconsumption = GrowFac_tp1*baseEx_inf.cFunc[0](m_tp1).T
        # finish expectation over perm shocks by right multiplying with weights
        part_expconsumption = np.dot(part_expconsumption, baseEx_inf.PermShkDstn[0].pmf)
        # finish expectation over trans shocks by right multiplying with prob weights
        expconsumption = np.dot(part_expconsumption, baseEx_inf.TranShkDstn[0].pmf)
        # return expected consumption
        return expconsumption

    # Calculate the expected consumption growth factor
    m1 = np.linspace(1,baseEx_inf.solution[0].mNrmSS,50) # m1 defines the plot range on the left of target m value (e.g. m <= target m)
    c_m1 = baseEx_inf.cFunc[0](m1)
    a1 = m1-c_m1
    exp_consumption_l1 = [exp_consumption(i) for i in a1]

    # growth1 defines the plotted values consumption growth factor when m is less than target m
    growth1 = np.array(exp_consumption_l1)/c_m1

    # m2 defines the plot range on the right of target m value (e.g. m >= target m)
    m2 = np.linspace(baseEx_inf.solution[0].mNrmSS,1.9,50)

    c_m2 = baseEx_inf.cFunc[0](m2)
    a2 = m2-c_m2
    exp_consumption_l2 = [exp_consumption(i) for i in a2]

    # growth 2 constructs values to plot of expected consumption growth factor when m is bigger than target m
    growth2 = np.array(exp_consumption_l2)/c_m2

    # Plot consumption growth as a function of market resources
    # Calculate Absolute Patience Factor Phi = lower bound of consumption growth factor
    AbsPatientFac = (baseEx_inf.Rfree*baseEx_inf.DiscFac)**(1.0/baseEx_inf.CRRA)

    fig = plt.figure(figsize = (12,8))
    ax = fig.add_subplot(111)
    # Plot the Absolute Patience Factor line
    ax.plot([0,1.9],[AbsPatientFac,AbsPatientFac],color="black")

    # Plot the Permanent Income Growth Factor line
    ax.plot([0,1.9],[baseEx_inf.PermGroFac[0],baseEx_inf.PermGroFac[0]],color="black")

    # Plot the expected consumption growth factor on the left side of target m
    ax.plot(m1,growth1,color="black", label='$\mathsf{E}_{t}[c_{t+1}/c_{t}]$')

    # Plot the expected consumption growth factor on the right side of target m
    ax.plot(m2,growth2,color="black", label='$\mathsf{E}_{t}[c_{t+1}/c_{t}]$')

    # Plot the arrows
    arrowplot(ax, m1,growth1)
    arrowplot(ax, m2,growth2, direc='pos')

    # Plot the target m
    ax.plot([baseEx_inf.solution[0].mNrmSS,baseEx_inf.solution[0].mNrmSS],[0,1.4],color="black",linestyle="--")
    ax.set_xlim(1,2.10)
    ax.set_ylim(0.98,1.08)
    ax.text(1,1.082,"Growth Rate",fontsize = 26,fontweight='bold')
    ax.text(2.105,0.975,"$m_{t}$",fontsize = 26,fontweight='bold')
    ax.text(baseEx_inf.solution[0].mNrmSS-0.02,0.974, r'$\check{m}$', fontsize = 26,fontweight='bold')
    ax.tick_params(labelbottom=False, labelleft=False,left='off',right='off',bottom='off',top='off')
    # ax.text(1.91,0.998,r'$\Phi = (\mathrm{\Rfree}\DiscFac)^{1/\CRRA}$',fontsize = 22,fontweight='bold')
    plt.legend()
    plt.show()
    # ax.text(1.91,1.03, r'$\PermGroFac$',fontsize = 22,fontweight='bold')

    