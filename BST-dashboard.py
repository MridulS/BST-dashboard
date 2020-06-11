# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: ExecuteTime,autoscroll,heading_collapsed,hidden,-hide_ouput,-code_folding
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.7
# ---

# %% [markdown]
# # Theoretical Foundations of Buffer Stock Saving
#
# [![econ-ark.org](https://img.shields.io/badge/Powered%20by-Econ--ARK-3e8acc.svg)](https://econ-ark.org/materials/BufferStockTheory)

# %%
from ipywidgets import interact, interactive, fixed, interact_manual
from dashboard_widget import (
    makeGICFailExample,
    makeConvergencePlot,
    DiscFac_widget,
    CRRA_widget,
    Rfree_widget,
    PermGroFac_widget,
    UnempPrb_widget,
    IncUnemp_widget,
    makeGrowthplot,
    makeBoundsfig,
    makeTargetMfig,
)
import HARK

HARK.logging.disable()
# The warnings package allows us to ignore some harmless but alarming warning messages
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Convergence of the Consumption Rules
#
# Under the given parameter values, [the paper's first figure](http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#Convergence-of-the-Consumption-Rules) depicts the successive consumption rules that apply in the last period of life $(c_{T}(m))$, the second-to-last period, and earlier periods $(c_{T-n})$.  $c(m)$ is the consumption function to which these converge:
#
# $$
# c(m) = \lim_{n \uparrow \infty} c_{T-n}(m)
# $$
#

# %%
a = interactive(
    makeConvergencePlot,
    DiscFac=DiscFac_widget[0],
    CRRA=CRRA_widget[0],
    Rfree=Rfree_widget[0],
    PermGroFac=PermGroFac_widget[0],
    UnempPrb=UnempPrb_widget[0],
)
a

# %% [markdown]
# ## [If the GIC Fails, Target Wealth is Infinite ](http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#The-GIC)
#
# [A figure](http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#FVACnotGIC) depicts a solution when the **FVAC** (Finite Value of Autarky Condition) and **WRIC** hold (so that the model has a solution) but the **GIC** (Growth Impatience Condition) fails.  In this case the target wealth ratio is infinity.
#
# The parameter values in this specific example are:
#
# | Param | Description | Code | Value |
# | :---: | ---         | ---  | :---: |
# | Γ | Permanent Income Growth Factor | $\texttt{PermGroFac}$ | 1.00 |
# | R | Interest Factor | $\texttt{Rfree}$ | 1.06 |
#

# %%
b = interactive(
    makeGICFailExample, Rfree=Rfree_widget[1], PermGroFac=PermGroFac_widget[1]
)
b


# %% [markdown]
# ### [Target $m$, Expected Consumption Growth, and Permanent Income Growth](https://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#AnalysisoftheConvergedConsumptionFunction)
#
# The next figure is shown in  [Analysis of the Converged Consumption Function](https://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#cGroTargetFig), which shows the expected consumption growth factor $\mathrm{\mathbb{E}}_{t}[c_{t+1}/c_{t}]$ for a consumer behaving according to the converged consumption rule.
#

# %%
c = interactive(
    makeGrowthplot,
    Rfree=Rfree_widget[2],
    PermGroFac=PermGroFac_widget[2],
    DiscFac=DiscFac_widget[2],
    CRRA=CRRA_widget[2],
)
c

# %% [markdown]
# ### [Consumption Function Bounds](https://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#AnalysisOfTheConvergedConsumptionFunction)
# [The next figure](https://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#cFuncBounds)
# illustrates theoretical bounds for the consumption function.
#
# We define two useful variables: lower bound of $\MPC$ (marginal propensity to consume) and limit of $h$ (Human wealth), along with some functions such as the limiting perfect foresight consumption function $\bar{c}(m)$, the upper bound function $\bar{\bar c}(m)$, and the lower bound function \underline{_c_}$(m)$.

# %%
d = interactive(
    makeBoundsfig,
    Rfree=Rfree_widget[3],
    PermGroFac=PermGroFac_widget[3],
    DiscFac=DiscFac_widget[3],
    CRRA=CRRA_widget[3],
)
d


# %% [markdown]
# ### [The Consumption Function and Target $m$](https://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#cFuncBounds)
#
# This figure shows the $\mathrm{\mathbb{E}}_{t}[\Delta m_{t+1}]$ and consumption function $c(m_{t})$, along with the intersection of these two functions, which defines the target value of $m$

# %%
e = interactive(
    makeTargetMfig,
    Rfree=Rfree_widget[4],
    PermGroFac=PermGroFac_widget[4],
    DiscFac=DiscFac_widget[4],
    CRRA=CRRA_widget[4],
)
e


# %%
