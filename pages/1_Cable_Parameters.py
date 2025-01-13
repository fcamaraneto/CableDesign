import streamlit as st

import pandas as pd
import numpy as np
import math
import io
import scipy.special as spios
#import scipy.io as spio
import plotly.express as px
import plotly.graph_objects as go
from scipy.linalg import block_diag
#from PIL import Image # create page icon

# Make some shortcuts
pi = np.pi
sqrt = np.sqrt
zeros = np.zeros
inverse = np.linalg.inv
transpose = np.transpose
conjugate = np.conjugate
identity = np.identity
abs = np.absolute
real = np.real
imag = np.imag
log = np.log
times = np.matmul
identity = np.identity

BesselJ = spios.jv
BesselI = spios.iv # gives the modified Bessel function of the first kind: equivalent a BesselI in Mathematica
BesselK = spios.kv # gives the modified Bessel function of the second kind: equivalent a BesselK in Mathematica

#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#                                     SETTINGS
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#icon=Image.open('dnv_logo.jpg')
#st.set_page_config(page_title="HELICA Multiphysics", layout="centered", page_icon=icon)

#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#                                     SIDEBAR
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://i.postimg.cc/wvSYBKsj/DNV-logo-RGB-Small.png);
                background-repeat: no-repeat;
                margin-left: 20px;
                padding-top: 120px;
                background-position: 1px 1px;
            }
            [data-testid="stSidebarNav"]::before {
              # content: "My Company Name";
              #  margin-left: 2px;
              #  margin-top: 2px;
              #  font-size: 3px;
              #  position: relative;
              #  top: 1px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
add_logo()


url='https://i.postimg.cc/NjhVmdYR/helica-logo.png'

st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] + div {{
                position:relative;
                bottom: 0;
                height:65%;
                background-image: url({url});
                background-size: 40% auto;
                background-repeat: no-repeat;
                background-position-x: center;
                background-position-y: bottom;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )





#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#                                     CROSS-SECTION
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
st.title("Cable Parameters")


#st.title("HELICA Current Rating")
#st.markdown('The Cable Rating module ... ')
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
tab1, tab2, tab3 = st.tabs(["🖥️ Input Data", "📊 Electrical Parameters", "🗂️ Export Results"])

with tab1:
    cable = st.selectbox("Select Cable Type",
                       options=["Single Core", "Three Core (future)"])
    #, "Pipe Type"



#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
#  1- Single Core (stranded sheath)
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

#    if cable == "Three Core (future)":
#        st.write("In development ...")


    if cable == "Single Core":
        #tubular = st.checkbox('Tubular sheath', key="disabled")
        tubular = False

        if tubular == False:

            #col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, .5, 1, .5, 1])
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, .7, .8])
            with col1:
                st.write("")
                radius1 = st.number_input('D1 [mm]', format="%.2f", value=30., step=1., min_value=.001)
                radius2 = st.number_input('D2 [mm]', format="%.2f", value=40., step=1., min_value=.001)
                radius3 = st.number_input('D3 [mm]', format="%.2f", value=50., step=1., min_value=.001)
            with col2:
                st.write("")
                radius4 = st.number_input('D4 [mm]', format="%.2f", value=65.51, step=1., min_value=.001)
                radius5 = st.number_input('D5 [mm]', format="%.2f", value=70.51, step=1., min_value=.001)
                radius6 = st.number_input('D6 [mm]', format="%.2f", value=74.51, step=1., min_value=.001)
            with col3: 
                st.write("")
                rcc = st.number_input('rc [mm]', format="%.2f", value=2., step=1., min_value=.001)
                rss = st.number_input('rs [mm]', format="%.2f", value=2., step=1., min_value=.001)
                raa = st.number_input('ra [mm]', format="%.2f", value=4., step=1., min_value=.001)
            with col4:
                st.write("")
                #nc = st.number_input('Nc ', value=2, step=1, min_value=1)
                ns = st.number_input('Ns', value=20, step=1, min_value=1)
                na = st.number_input('Na', value=50, step=1, min_value=1)
            with col5:
                st.write("")
                Np = st.number_input('Nfourier', value=1, step=1, min_value=0)
                #Np = 1
                nf = st.number_input('Samples', value=30, step=1, min_value=0)

            D1 = radius1 * 1.e-3
            D2 = radius2 * 1.e-3
            D3 = radius3 * 1.e-3
            D4 = radius4 * 1.e-3
            D5 = radius5 * 1.e-3
            D6 = radius6 * 1.e-3
            rc = rcc * 1e-3;
            rs = rss * 1e-3;
            ra = raa * 1e-3;
            
            outc = D1
            theta_s = 360/ns
            theta_a = 360/na

            if (D1 == rc):
                layers = 0
                nc = [1]
                theta_c = [0]
                R1c = [0]
            else:
                layers = int(np.floor(((outc + 1.e-5) / rc) - np.floor(0.5 * (outc + 1.e-5) / rc)) - 1)
                nc = np.zeros(layers)
                nc = [1] + [(i * 6) for i in range(1, layers + 1)]
                theta_c = [0] + [(360 / nc[i]) for i in range(1, layers + 1)]
                R1c = [2 * rc * i for i in range(0, layers + 1)]

            xc = np.zeros(sum(nc), dtype='float32')
            yc = np.zeros(sum(nc), dtype='float32')

            for k in range(0, layers + 1):
                a = sum(nc[0:k])
                b = sum(nc[0:k + 1])

                xc[a:b] = [R1c[k] * np.cos(i * (theta_c[k] * np.pi / 180)) for i in range(1, nc[k] + 1)]
                yc[a:b] = [R1c[k] * np.sin(i * (theta_c[k] * np.pi / 180)) for i in range(1, nc[k] + 1)]

            xs = [D3 * np.cos(i * (theta_s * np.pi / 180)) for i in range(0, ns)]
            ys = [D3 * np.sin(i * (theta_s * np.pi / 180)) for i in range(0, ns)]
            xa = [D5 * np.cos(i * (theta_a * np.pi / 180)) for i in range(0, na)]
            ya = [D5 * np.sin(i * (theta_a * np.pi / 180)) for i in range(0, na)]

            x = np.concatenate((xc, xs, xa))
            y = np.concatenate((yc, ys, ya))

            # distancia entre condutores
            d = np.zeros((len(x), len(x)))

            for i in range(len(x)):
                for j in range(len(x)):
                    d[i, j] = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

            #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            #                      PLOT cross-section (1core)
            #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            fig = go.Figure()
            shapes = []

            # CORE
            for i in range(len(xc)):
                shapes.append(dict(type="circle", xref="x", yref="y",
                                   x0=xc[i] - 1 * rc, y0=yc[i] - 1 * rc,
                                   x1=xc[i] + 1 * rc, y1=yc[i] + 1 * rc,
                                   line_color="LightSeaGreen"))
            shapes.append(dict(type="circle", xref="x", yref="y",
                               x0= min(xc) - 1 * rc, y0= -max(xc) - 1 * rc,
                               x1= max(xc) + 1 * rc, y1= max(xc) + 1 * rc,
                               line_color="LightSeaGreen"))

            # SHEATH
            for i in range(ns):
                shapes.append(dict(type="circle", xref="x", yref="y",
                                   x0= xs[i] - 1 * rs, y0= ys[i] - 1 * rs,
                                   x1= xs[i] + 1 * rs, y1= ys[i] + 1 * rs,
                                   line_color="LightSeaGreen")) 
                shapes.append(dict(type="circle", xref="x", yref="y",
                                   x0=min(xs) + 1 * rs, y0=-max(xs) + rs,
                                   x1=max(xs) - 1 * rs, y1=max(xs) - rs,
                                   line_color="LightSeaGreen"))
                shapes.append(dict(type="circle", xref="x", yref="y",
                                   x0=min(xs) - 1 * rs, y0=-max(xs) - 1 * rs,
                                   x1=max(xs) + 1 * rs, y1=max(xs) + 1 * rs,
                                   line_color="LightSeaGreen"))

            # ARMOR
            for i in range(na):
                shapes.append(dict(type="circle", xref="x", yref="y",
                                 x0=xa[i] - ra, y0=ya[i] - 1 * ra,
                                   x1=xa[i] + ra, y1=ya[i] + 1 * ra,
                                   line_color="LightSeaGreen"))
            for i in range(na):
                shapes.append(dict(type="circle", xref="x", yref="y",
                                   x0=min(xa) + ra, y0=min(ya) + 1 * ra,
                                   x1=max(xa) - ra, y1=max(ya) - 1 * ra,
                                   line_color="LightSeaGreen"))
                shapes.append(dict(type="circle", xref="x", yref="y",
                                   x0=min(xa) - ra, y0=min(ya) - 1 * ra,
                                   x1=max(xa) + ra, y1=max(ya) + 1 * ra,
                                   line_color="LightSeaGreen"))

            # OUTER COVER
            shapes.append(dict(type="circle", xref="x", yref="y",
                               x0 = -D6, y0 = -D6,
                               x1 = D6, y1 = D6,
                               line_color="LightSeaGreen"))



            col1, col2, col3 = st.columns([.3, 1, .2])
            with col2:
                fig.update_layout(width=500, height=500)
                fig.update_xaxes(range=[-max(x) * 1.25, max(x) * 1.25])
                fig.update_yaxes(range=[-max(x) * 1.25, max(x) * 1.25])
                fig.update_xaxes(visible=False, mirror=True, ticks='outside', showline=True, linecolor='black',
                                 gridcolor='white')
                fig.update_yaxes(visible=False, mirror=True, ticks='outside', showline=True, linecolor='black',
                                 gridcolor='white')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                fig.update_layout(shapes=shapes)
                st.plotly_chart(fig)




        else:
            ''

    else:
        #st.warning('Implementation in development.')
        st.success('Implementation in development.')
        st.stop()


            #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            #  2- Single Core (tubular sheath)
            #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            #if cable2 == "Single Core (tubular sheath)"


        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        #  3- Three Core (stranded sheath)
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            #if cable2 == "Three Core":



#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
# MoM-SO: ELECTRICAL PARAMETERS
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
with tab2:
    # frequency range
    f = np.logspace(-2, 6, nf)
    s = 2*pi*f

    # truncated Fourier series order
    nn = 2*Np + 1

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #   Green's Functions
    def f45(dpq):
        sol = np.log(dpq) / (2 * pi)
        return (sol)

    def f46(ap, dpq, n2, xpq, ypq):
        sol1 = (-1 / (4 * pi * abs(n2))) * np.power(ap / dpq, abs(n2))
        sol2 = np.power(-(xpq - 1j * ypq) / dpq, n2)
        sol = sol1 * sol2
        return (sol)

    def f50(ap, aq, n2, n, xpq, ypq):
        sol1 = (-pi * np.power(aq, n) / np.power(-ap, n2)) * spios.binom(n - n2 - 1, -n2)
        sol2 = np.power(xpq - 1j * ypq, -n + n2) / (np.power(2 * pi, 2) * n)
        sol = sol1 * sol2
        return (sol)

    def f52(radius):
        sol = np.log(radius) / (2 * pi)
        return (sol)

    def f53(order):
        sol = -1 / (4 * pi * abs(order))
        return (sol)

    def gij(r, k, m, x, y, d, Np):
        n = 2 * Np + 1
        g = zeros((n, n), dtype=complex)
        for nm in range(n + 1):
            if nm != Np + 1:
                g[nm - 1, Np] = f46(r[k], d[k, m], -Np - 1 + nm, x[k] - x[m], y[k] - y[m])
            else:
                g[nm - 1, Np] = f45(d[k, m])
        for i in range(1, Np + 2):
            for j in range(Np + 1, 2 * Np + 1):
                g[i - 1, j] = f50(r[k], r[m], -Np - 1 + i, j - Np, x[k] - x[m], y[k] - y[m])
        for i in range(n):
            g[i, 0:Np] = conjugate(np.flip(g[(2 * Np) - i, Np + 1:2 * Np + 1]))
        return g
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

    # CONSTANTS
    mu = 4 * pi * 1e-7;
    eps = 8.854e-12

    # condutor
    mu_c = 1
    mu_s = 1
    mu_a = 1

    eps_c = 1
    eps_s = 1
    eps_a = 1

    sig_c = 5.8e7
    sig_s = 5.8e7
    sig_a = .58e7

    rho_c = 1 / sig_c
    rho_s = 1 / sig_s
    rho_a = 1 / sig_a

    # meio isolante
    mu_isol1 = 1.
    mu_isol2 = 1.
    mu_isol3 = 1.

    eps_isol1 = 2.5
    eps_isol2 = 2.5
    eps_isol3 = 2.5

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #   SURFACE OPERATOR
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    def ysolid(Np, omega, radius, mur1, mur2, epsr1, epsr2, sigma):
        mu = 4 * pi * 1e-7
        eps = 8.854e-12

        sol = zeros((2 * Np + 1, 2 * Np + 1), dtype=complex)

        k = sqrt(omega * mur1 * mu * (omega * eps * epsr1 - 1j * sigma))
        k0 = omega * sqrt(mur2 * mu * eps * epsr2)

        for np in range(-Np, Np + 1):
            bessel = BesselJ(np, k * radius)
            bessel0 = BesselJ(np, k0 * radius)
            dbessel = 0.5 * (BesselJ(-1 + np, k * radius) - BesselJ(1 + np, k * radius))
            dbessel0 = 0.5 * (BesselJ(-1 + np, k0 * radius) - BesselJ(1 + np, k0 * radius))

            sol[np + Np, np + Np] = (2 * pi / (1j * omega)) * (
                        (k * radius * dbessel / (mu * mur1 * bessel)) - k0 * radius * dbessel0 / (mu * mur2 * bessel0))

        return (sol)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #   G MATRIX
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    Nc = len(xc)  # core: number of condutors
    Ns = len(xs)  # sheath: number of condutors
    Na = len(xa)  # armour1: number of condutors
    #Na2 = len(xa2) # armour2: number of condutors
    #Na = Na1  # +Na2 # armour: total number of condutors
    n = len(np.concatenate((xc, xs, xa)))  # , xa2))) # total number of condutors

    Rc = [rc for _ in range(Nc)]
    Rs = [rs for _ in range(Ns)]
    Ra = [ra for _ in range(Na)]
    # Ra2 = [ra for _ in range(Na2)]
    r = np.concatenate((Rc, Rs, Ra))  # , Ra2))

    N1 = 2*Np + 1  # ordem da matriz g:-> (nxn)
    N2 = (2*Np + 1)*n  # ordem da matriz G : -> (NxN)

    gc = np.matrix(zeros((N1, N1), dtype=complex))
    gs = np.matrix(zeros((N1, N1), dtype=complex))
    ga = np.matrix(zeros((N1, N1), dtype=complex))

    G = np.matrix(zeros((N2, N2), dtype=complex))
    Gdiag = np.matrix(zeros((N2, N2), dtype=complex))
    Goff = np.matrix(zeros((N2, N2), dtype=complex))

    # g_core
    for nm in range(1, N1 + 1):
        if nm != Np + 1:
            gc[nm - 1, nm - 1] = f53(-Np - 1 + nm)
        else:
            gc[nm - 1, nm - 1] = f52(rc)

    # g_sheath
    for nm in range(1, N1 + 1):
        if nm != Np + 1:
            gs[nm - 1, nm - 1] = f53(-Np - 1 + nm)
        else:
            gs[nm - 1, nm - 1] = f52(rs)

    # g_armor
    for nm in range(1, N1 + 1):
        if nm != Np + 1:
            ga[nm - 1, nm - 1] = f53(-Np - 1 + nm)
        else:
            ga[nm - 1, nm - 1] = f52(ra)

    # Gdiag (diagonal elements)
    for i in range(Nc):
        Gdiag[i * (2 * Np + 1):(i + 1) * (2 * Np + 1), i * (2 * Np + 1):(i + 1) * (2 * Np + 1)] = gc

    for i in range(Nc, Nc + Ns):
        Gdiag[i * (2 * Np + 1):(i + 1) * (2 * Np + 1), i * (2 * Np + 1):(i + 1) * (2 * Np + 1)] = gs

    for i in range(Nc + Ns, Nc + Ns + Na):
        Gdiag[i * (2 * Np + 1):(i + 1) * (2 * Np + 1), i * (2 * Np + 1):(i + 1) * (2 * Np + 1)] = ga

    # Goff (off-diagonal elements)
    for i in range(n):
        for j in range(n):
            if j > i:
                Goff[i * (2 * Np + 1):(i + 1) * (2 * Np + 1), j * (2 * Np + 1):(j + 1) * (2 * Np + 1)] = gij(r, i, j, x,
                                                                                                             y, d, Np)
    G = Gdiag + Goff + Goff.getH()


    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #   LOOP (INITIALIZATION)
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    u1 = zeros(2 * Np + 1)
    for i in range(-Np, Np + 1):
        if i == 0:
            u1[i + Np] = 1

    U = zeros((N2, n))
    for i in range(n):
        U[i * (2 * Np + 1): (i + 1) * (2 * Np + 1), i] = u1

    MetLayers = 3
    Q = zeros((MetLayers, n))
    for i in range(Nc):
        Q[0, :Nc] = 1
    for i in range(Nc, Nc + Ns):
        Q[1, Nc:Nc + Ns] = 1
    for i in range(Nc + Ns, Nc + Ns + Na):
        Q[2, Nc + Ns:] = 1

    S = zeros((3, 2))
    S[0, :] = [1, 0]
    S[1, :] = [-1, -1]
    S[2, :] = [0, 1]

    Z = zeros((nf, 2), dtype=complex)
    L = zeros((nf, 2))
    R = zeros((nf, 2))
    Zint = zeros((nf,3,3), dtype=complex)
    zint = zeros((nf,5), dtype=complex)
    Yint = zeros((nf,3,3), dtype=complex)
    yint = zeros((nf,3), dtype=complex)

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #  MAIN LOOP
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    for nm in range(nf):
        omega = 2 * pi * f[nm]

        ySO1 = ysolid(Np, omega, rc, mu_c, mu_isol1, eps_c, eps_isol1, sig_c)
        ySO2 = ysolid(Np, omega, rs, mu_s, mu_isol2, eps_s, eps_isol2, sig_s)
        ySO3 = ysolid(Np, omega, ra, mu_a, mu_isol3, eps_a, eps_isol3, sig_a)

        Yc = zeros((Nc * (2 * Np + 1), Nc * (2 * Np + 1)), dtype=complex)
        Ys = zeros((Ns * (2 * Np + 1), Ns * (2 * Np + 1)), dtype=complex)
        Ya = zeros((Na * (2 * Np + 1), Na * (2 * Np + 1)), dtype=complex)

        for i in range(Nc):
            Yc[i * (2 * Np + 1):(i + 1) * (2 * Np + 1), i * (2 * Np + 1):(i + 1) * (2 * Np + 1)] = ySO1

        for i in range(Ns):
            Ys[i * (2 * Np + 1):(i + 1) * (2 * Np + 1), i * (2 * Np + 1):(i + 1) * (2 * Np + 1)] = ySO2

        for i in range(Na):
            Ya[i * (2 * Np + 1):(i + 1) * (2 * Np + 1), i * (2 * Np + 1):(i + 1) * (2 * Np + 1)] = ySO3

        Y = block_diag(Yc, Ys, Ya)

        dum1 = times(Y, U)
        dum2 = inverse((identity(N2) - 1j * omega * mu * times(Y, G)))
        dum3 = times(dum2, dum1)
        dum4 = times(transpose(U), dum3)
        zz = inverse(dum4)

        dum5 = times(inverse(zz), transpose(Q))
        dum6 = inverse(times(Q, dum5))
        dum7 = times(dum6, S)
        z = times(transpose(S), dum7)

        partialZ = inverse(times(Q, times(inverse(zz), transpose(Q))))
        zloop1 = partialZ[0, 0] + partialZ[1, 1] - 2 * partialZ[0, 1]
        zloop2 = partialZ[1, 1] + partialZ[2, 2] - 2 * partialZ[1, 2]
        Z[nm, :] = [zloop1, zloop2]
        R[nm, :] = [real(zloop1), real(zloop2)]
        L[nm, :] = [imag(zloop1) / omega, imag(zloop2) / omega]

        # ANALYTICAL EXPRESSIONS

        eta_s = sqrt(1j * omega * mu_s * mu / rho_s)
        eta_a = sqrt(1j * omega * mu_a * mu / rho_a)

        # SHEATH
        # z2m: sheath mutual impedance
        dum1 = (1 / (BesselI(1, eta_s * D3) * BesselK(1, eta_s * D2) - BesselI(1, eta_s * D2) * BesselK(1, eta_s * D3)))
        z2m = (rho_s / (2 * pi * D2 * D3)) * dum1

        # ARMOUR
        # z3m: armour mutual impedance
        dum2 = (1 / (BesselI(1, eta_a * D5) * BesselK(1, eta_a * D4) - BesselI(1, eta_a * D4) * BesselK(1, eta_a * D5)))
        z3m = (rho_a / (2 * pi * D4 * D5)) * dum2

        # ARMOUR
        # z30 : internal impedance of armour outer surface
        dum3 = BesselI(0, eta_a * D5) * BesselK(1, eta_a * D4) + BesselK(0, eta_a * D5) * BesselI(1, eta_a * D4)
        dum4 = BesselI(1, eta_a * D5) * BesselK(1, eta_a * D4) - BesselI(1, eta_a * D4) * BesselK(1, eta_a * D5)
        z30 = (rho_a * eta_a / (2 * pi * D5)) * dum3 / dum4

        # OUTER  COVER
        # z34:sheath outer insulator impedance
        z34 = 1j * (omega * mu / (2 * pi)) * log(D6 / D5)

        Zcc = zloop1 - 2 * z2m + zloop2 + z30 + z34 - 2 * z3m;
        Zss = zloop2 + (z30 + z34) - 2 * z3m;
        Zaa = z30 + z34;

        Zcs = zloop2 - z2m - 2 * z3m + z30 + z34;
        Zca = -z3m + z30 + z34;
        Zsa = -z3m + z30 + z34;

        Zint[nm, :, :] = [
            [Zcc, Zcs, Zca],
            [0, Zss, Zsa],
            [0, 0, Zaa]]

        zint[nm,:] = [Zcc, Zss, Zaa, Zcs, Zsa]

        # ADMITTANCE MATRIX 

        y1 = (1j * omega * 2 * pi) * (eps_isol1 * eps / log(D2 / D1))
        y2 = (1j * omega * 2 * pi) * (eps_isol2 * eps / log(D4 / D3))
        y3 = (1j * omega * 2 * pi) * (eps_isol3 * eps / log(D6 / D5))

        Yint[nm, :, :] = [
            [y1, -y1, 0],
            [-y1, y1 + y2, -y2],
            [0, -y2, y2 + y3]]

        yint[nm, :] = [y1, y1 + y2, y2 + y3]

#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
# EXPORT PSCAD
#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -



    #PSCADfile(f, Zpul, Ypul, 2.5, "UCC")

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # export parameters
    #np.savetxt('R_webapp.dat', R, fmt='%.16f', delimiter='\t')
    #np.savetxt('L_webapp.dat', L, fmt='%.16f', delimiter='\t')
    #import scipy.io
    #scipy.io.savemat('zint_webapp.mat', {'freq': f, 'data': Zint})
    #scipy.io.savemat('yint_python.mat', {'freq':f, 'data': Yint})

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # DATAFRAME
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    idx = f
    columnsZ = ['Z11', 'Z22', 'Z33', 'Z12', 'Z23']
    columnsY = ['Y11', 'Y22', 'Y33']
    df1 = pd.DataFrame(abs(zint), index=idx, columns=columnsZ)
    df2 = pd.DataFrame(real(zint), index=idx, columns=columnsZ)
    df3 = pd.DataFrame(imag(zint), index=idx, columns=columnsZ)
    df4 = pd.DataFrame(abs(yint), index=idx, columns=columnsY)

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #              PLOT
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    st.markdown("""### Impedance Matrix""")
    fig = px.line(df1, log_x=True, log_y=True)
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Abs@Z (Ω)")
    fig.update_layout(legend_title="Zpul (Ω)")
    fig.update_xaxes(exponentformat="SI")
    fig.update_yaxes(exponentformat="e")  # "SI"
    fig.update_layout(width=900, height=600)
    st.plotly_chart(fig, use_container_width=False)

    st.markdown("""### Admittance Matrix""")
    fig4 = px.line(df4, log_x=True, log_y=True)
    fig4.update_xaxes(title_text="Frequency (Hz)")
    fig4.update_yaxes(title_text="Abs@Y (S)")
    fig4.update_layout(legend_title="Ypul (S)")
    fig4.update_xaxes(exponentformat="SI")
    fig4.update_yaxes(exponentformat="e")  # "SI"
    fig4.update_layout(width=900, height=600)
    st.plotly_chart(fig4, use_container_width=False)

    #st.markdown("""### Impedance Matrix (Re)""")
    #fig2 = px.line(df2, log_x=True, log_y=True)
    #fig2.update_xaxes(title_text="Frequency (Hz)")
    #fig2.update_yaxes(title_text="Re@Z (Ω)")
    #fig2.update_layout(legend_title="Re@Z (Ω)")
    #fig2.update_xaxes(exponentformat="SI")
    #fig2.update_yaxes(exponentformat="e")  # "SI"
    #fig2.update_layout(width=900, height=600)
    #st.plotly_chart(fig2, use_container_width=False)

    #st.markdown("""### Impedance Matrix (Im)""")
    #fig3 = px.line(df3, log_x=True, log_y=True)
    #fig3.update_xaxes(title_text="Frequency (Hz)")
    #fig3.update_yaxes(title_text="Im@Z (Ω)")
    #fig3.update_layout(legend_title="Im@Z (Ω)")
    #fig3.update_xaxes(exponentformat="SI")
    #fig3.update_yaxes(exponentformat="e")  # "SI"
    #fig3.update_layout(width=900, height=600)
    #st.plotly_chart(fig3, use_container_width=False)







#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
with tab3:

    import time
    # Define the function to create the text data
    def create_pscad_text(f, Z, Y, length):
        nf = Z.shape[0]
        dim = Z.shape[1]
        text_data = "! " + time.strftime("Date:%d-%m-%Y  Time:%H:%M:%S", time.localtime()) + "\n"
        text_data += "! Cable data generated by an external program\n"
        localtime = time.asctime(time.localtime(time.time()))
        text_data += f"{nf}\n"
        text_data += f"{dim}\n"
        for nm in range(nf):
            text_data += f"{f[nm]}\n"
            for i in range(dim):
                for j in range(dim):
                    text_data += f"{np.real(Z[nm, i, j]):.6e}\n"
                    text_data += f"{np.imag(Z[nm, i, j]):.6e}\n"
                    text_data += f"{np.real(Y[nm, i, j]):.6e}\n"
                    text_data += f"{np.imag(Y[nm, i, j]):.6e}\n"
        return text_data


    st.subheader('Interface with EMT-type solver')
    #st.markdown(' Interfacing with circuit solvers contains matlab scripts which demonstrate'
    #        ' how to interface rational function-based models with time domain circuit solvers '
    #        'via a Norton equivalent. The procedure is shown for models representing '
    #        'Y-parameters, Z-parameters, S-parameters, and general transfer functions that '
    #        'do not interact with the circuit.')

    col1, col2, col3, col4 = st.columns([1.5, 1, 2, 1.25])
    with col1:
        st.write("")
        st.selectbox("Select Software:",
                     options=["PSCAD", "ATP (future)", "EMTP (future)", "PowerFactory (future)"])
    with col2:
        st.write("")
        length = st.number_input('Cable length (km):', value=1., min_value=0.001)
    with col3:
        st.write("")
        filename = st.text_input("YZ Output File Name (.txt):", "ZYpscad")
    with col4:
        st.write("")
        st.write("")
        st.write("")
        text_data = create_pscad_text(f, Zint, Yint, length)
        text_bytes = text_data.encode('utf-8')
        text_binary_buffer = io.BytesIO(text_bytes)

        # Initialize the session state for button click
        if 'button_clicked' not in st.session_state:
            st.session_state['button_clicked'] = False
        # Define the button label based on the session state
        button_label = "Download" if not st.session_state['button_clicked'] else "Downloaded ✅"
        # Create a download button with the dynamic label

        btn = st.download_button(
            label=button_label,
            data=text_binary_buffer,
            file_name=filename+".txt",
            mime="text/plain",
            on_click=lambda: setattr(st.session_state, 'button_clicked', True)
        )

        # Update the session state when the button is clicked
        if btn:
            st.session_state['button_clicked'] = True

        

        #btn = st.download_button(
        #    label="Save ✔️",
        #    data=text_binary_buffer,
        #    file_name=filename+".txt",
        #    mime="text/plain")
        #if btn:
        #    st.write('Done!')




    st.write("")

    #PSCADfile(f, Zint, Yint, 2.5, "UCC")

    #st.download_button(
        #label="Download Parameters",
        #data= PSCADfile(f, Zint, Yint, 2.5, "UCC"),
        #data= 'Universal Cable Constants (UCC) \n\n' + f,
        #file_name='cable_parameters.txt',
        #mime='text')

    #st.download_button(
    #    label="Download Report",
    #    data='Universal Cable Constants (UCC) \n\n' + dum,
    #    file_name='cable_parameters.txt',
    #    mime='text/csv')




    st.write("")
