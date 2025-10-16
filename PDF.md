Defocus-Aware Modeling and Control Analysis
of a QD-Based Optical Tracking System:
Experimental and Simulated Evaluation
Using the DOLCE Terminal
Hideki Takamoto
Department of Aeronautics and
Astronautics
The University of Tokyo
Tokyo, Japan
takamoto@space.t.u-tokyo.ac.jp
Kuna Shitara
Department of Aeronautics and
Astronautics
The University of Tokyo
Tokyo, Japan
shitara@space.t.u-tokyo.ac.jp
Vinicius Ferreira Nery
Department of Aeronautics and
Astronautics
The University of Tokyo
Tokyo, Japan
viniciusfnery@space.t.u-tokyo.ac.jp
Kazuki Takashima
Department of Aeronautics and
Astronautics
The University of Tokyo
Tokyo, Japan
takashima@space.t.u-tokyo.ac.jp
Yuki Kusano
Department of Aeronautics and
Astronautics
The University of Tokyo
Tokyo, Japan
kusano@space.t.u-tokyo.ac.jp
Norihide Miyamura
School of Science and Engineering
Meisei University
Tokyo, Japan
norihide.miyamura@meisei-u.ac.jp
Kota Kakihara
Arkedge Space Inc.
Tokyo, Japan
kakihara-kota@arkedgespace.com
Toshihiro Suzuki
Department of Aeronautics and
Astronautics
The University of Tokyo
Tokyo, Japan
suzuki t@space.t.u-tokyo.ac.jp
Takayuki Hosonuma
JAXA
Kanagawa, Japan
hosonuma.takayuki@gmail.com
Satoshi Ikari
Department of Aeronautics and
Astronautics
The University of Tokyo
Tokyo, Japan
ikari@space.t.u-tokyo.ac.jp
Ryu Funase
Department of Aeronautics and
Astronautics
The University of Tokyo
Tokyo, Japan
funase@space.t.u-tokyo.ac.jp
Shinichi Nakasuka
Department of Aeronautics and
Astronautics
The University of Tokyo
Tokyo, Japan
nakasuka@space.t.u-tokyo.ac.jp
Abstract—Free-space optical communication (FSOC) is at-
tracting significant attention as a core technology for future space
communication infrastructure, owing to its capability of provid-
ing wide bandwidth and high antenna gain with narrow beam
divergence. However, the small divergence angle of optical beams
makes the link highly sensitive to pointing errors, requiring
highly accurate tracking sensors. The quadrant detector (QD),
with its simple structure and fast response, is a strong candidate
for fine tracking, but its linear response range is severely limited
when placed at the focal plane. To address this limitation, this
study investigates a defocus-based approach in which the QD
is intentionally placed out of focus to expand its field of view.
Using wave-optics simulations and laboratory experiments, the
response characteristics of QDs under defocus conditions are
systematically evaluated. Furthermore, acquisition and tracking
experiments were conducted with a numerical terminal simulator
to verify the validity of QD defocus. The results clarify the
trade-off between expanded linear range and tracking accuracy,
providing valuable insights for the design of compact and cost-
effective optical communication terminals.
Index Terms—Free-space optical communication (FSOC),
Pointing, Acquisition, and Tracking (PAT), Quadrant Detector
(QD), Defocus, Terminal Simulator
I. BACKGROUND
In recent years, driven by the growing demand for high-
capacity data transmission between satellites and between
satellites and the ground, research and development of free-
space optical communication (FSOC) has been actively pur-
sued [1]. Compared with microwave communication, optical
communication offers significant advantages such as a wider
bandwidth and higher antenna gain resulting from its narrow
beam divergence [1], making it a promising core technology
for next-generation space communication infrastructure. How-
ever, due to the small divergence angle of optical beams, the
system is highly sensitive to even slight attitude errors of the
transmitter and receiver as well as to their relative motion.
Therefore, the implementation of highly accurate pointing,
acquisition, and tracking (PAT) functions is indispensable [1].
In many previous studies, fine tracking within PAT systems
has employed sensors such as camera-based imagers and
position-sensitive detectors (PSDs). Among these, the quadrant
detector (QD) has attracted particular attention as a candidate
sensor for spaceborne optical communication terminals, owing
to its simple structure and fast response. A QD consists of
a photodiode divided into four quadrants, where the incident
spot position can be estimated by computing differential ratios
of photocurrent outputs from the quadrants. This enables fast
and low-latency acquisition of angular error signals, which is
a major advantage for compensating attitude fluctuations and
platform vibrations.
One of the major limitations of the QD, however, is its
extremely narrow linear response range. When the QD is
placed at the focal plane, the region over which the output
signal maintains an approximately proportional relationship is
constrained by the diffraction limit of the lens, and typically
extends to only several tens of microradians. Consequently, its
effective operating range is limited in situations requiring wide
fields of view (FOV), such as initial acquisition or cases with
large platform pointing errors. Conventional optical terminal
designs have addressed this issue by combining coarse tracking
sensors with wide FOV and fine tracking sensors with high
precision but narrow FOV. Nevertheless, for future optical
communication systems aiming at miniaturization and cost
reduction, simplification of sensor configuration and relaxation
of design constraints are strongly desired.
To address this challenge, the present study focuses on the
“defocus” method, in which the QD is intentionally placed out
of the focal plane. In such a configuration, the spot size ex-
pands and the linear response range has the potential to extend.
At the same time, however, the intensity distribution deviates
from a simple flat-spot approximation or Airy-disk profile
and becomes nonlinear. Most previous studies have relied
on idealized distribution models, and detailed evaluations of
QD response characteristics under actual defocused conditions
have been limited. Therefore, accurately characterizing the
QD response under defocus is essential to provide important
insights for designing tracking systems that aim to achieve
both wide FOV and high accuracy.
The objective of this study is to systematically evaluate
the sensitivity characteristics and linear response range of
QDs under defocus conditions through theoretical modeling,
numerical simulation, and experimental validation. In the sim-
ulation, a wave-optics-based propagation model is employed
to generate spot intensity distributions for various defocus
distances, which are then applied to calculate the QD voltage
outputs, thereby constructing a nonlinear QD response model.
In the experimental study, a QD is mounted on an optical
bench, and the relative distance from the focal plane is varied
while injecting the optical spot, so that the response variation
with defocus can be directly measured. These experiments
validate the simulation model and quantitatively assess the
extent of linear FOV expansion under defocus. Furthermore,
PAT experiments were conducted using a numerical termi-
nal simulator to verify the effectiveness of the QD defocus
method.
Originally, the research plan included conducting tracking
control experiments using a QD within the DOLCE terminal.
However, due to constraints in research resources and avail-
able experiment time, the present work focuses on defocus
experiments with a standalone QD, emphasizing its output
characteristics and their correlation with the simulation model.
Although control experiments are limited to simulator analysis,
a deeper understanding of the QD sensor behavior is expected
to provide design guidelines that can be extended to full
terminal systems in the future.
In summary, this study comprehensively analyzes QD defo-
cus behavior from both optical system and sensor response per-
spectives, thereby providing fundamental knowledge toward
the simultaneous realization of wider FOV and high-precision
tracking performance in optical communication terminal de-
sign.
II. PRINCIPLE OF QD
Fig. 1: Schematic of a QD-based fine tracking sensor
A QD consists of four photodiodes arranged in a quadrant
configuration. The photocurrents generated in each segment
are converted into voltages by operational amplifiers, allowing
the incident beam position to be detected. Compared to other
optical sensors, the QD is characterized by its high angular
resolution and fast response speed. Figure 1 illustrates a typical
configuration when a QD is employed as a fine-tracking sensor.
Here, the focal length of the front lens is denoted as f , and
the defocus, defined as the distance from the focal plane to
the detector surface, is denoted as d.
As shown in the figure, the incident beam is focused by
the front lens and directed onto the QD, which is placed
near the focal plane. When the incident beam deviates in
angle, the spot position on the QD shifts accordingly. This
displacement changes the ratio of the optical power received
by the four quadrants, from which the beam spot position
can be estimated. When the spot is located near the center of
the QD, the output signals of the quadrants are approximately
proportional to the displacement of the spot from the center.
If we denote the output voltages corresponding to the four
quadrants A–D as va–vd, the beam spot position (px, py ) can
be expressed as follows [2]:
px = P0
(va + vd) − (vb + vc)
va + vb + vc + vd
, (1)
py = P0
(va + vb) − (vc + vd)
va + vb + vc + vd
(2)
Here, P0 represents the region in which the QD provides a
linear response as a tracking sensor, referred to as the linear
field of view (FOV). When the spot on the QD is assumed to
be a circular flat-top spot with diameter ds, simple geometric
analysis shows that the proportionality constant is given by
Kf lat = π/8 ≃ 0.393, i.e.,
P0 = Kf lat · ds = π
8 ds (3)
On the other hand, when the QD is placed exactly at the
focal plane of the lens, the tracking accuracy is ultimately
limited by the diffraction condition. In this case, the spot
forms an Airy distribution, and using the Airy disk diameter
(first dark ring), defined as dA = 1.22λf /R (where R is the
aperture radius and λ is the wavelength), P0 can be expressed
as [3]:
P0 = 3π
32 · λf
R = 1
4.14 dA (4)
In practice, however, when the QD is placed away from
the focal plane, the spot does not become flat but exhibits a
non-uniform intensity distribution. Thus, the response does not
fully agree with either of the above expressions. To address
this, we define a conversion coefficient K under defocused
conditions, and P0 is expressed in terms of the geometric spot
diameter ds as:
P0 = K · ds (5)
Finally, the beam directions (qx, qy ) are obtained by dividing
the estimated spot positions by the focal length of the lens.
qx = px
f , qy = py
f (6)
III. QD NUMERICAL SIMULATOR
A. Analysis Method
In this section, we describe a numerical simulator developed
to model the output characteristics of a QD under various
spot conditions. The simulator calculates the relative output
voltages of each quadrant cell by summing the optical intensity
incident on each cell when the spot image is placed at a
given position on the QD surface. By applying the position
estimation algorithm described in the previous section to these
simulated outputs, the behavior of the linear FOV as well as
the estimated spot position can be evaluated. All parameter
values are set identical to those used in the experimental
configuration presented in Table 2.
Fig. 2: Simulated spot intensity distributions under different
defocus conditions.
In the simulation experiment, the first step is to generate
the intensity distributions of the spot under different defocus
conditions. At zero defocus, the spot follows an Airy distribu-
tion, while under defocused conditions, the spot does not form
a flat-top distribution but instead exhibits a specific intensity
profile.
Next, for each spot distribution, the simulator computes the
cell output voltages of quadrants A–D as the spot is shifted
along the QD sensor’s X-axis, as illustrated in Fig. 3. The
position estimation algorithm is then applied to the calculated
outputs. This procedure enables the evaluation of both the
estimation accuracy and the extent of the linear FOV.
Fig. 3: Illustration of spot displacement across the QD sensor
in the simulation.
B. Simulation Results
This section presents representative outputs from the QD
simulator and discusses the estimation of the optimal conver-
sion coefficient K under defocused conditions. The aim is
to achieve position estimation with higher accuracy than that
provided by the conventional flat-spot assumption.
Fig. 4 presents examples of the QD output voltage ratios as
a function of the spot center position for several defocus condi-
tions. The simulated responses clearly reproduce the behavior
near the linear FOV as well as the influence of noise at low
received optical power. The modeled noise sources include
dark noise, shot noise, and amplifier noise. As the defocus
increases, the spot size becomes larger, resulting in a shallower
slope in the response curves and an expanded linear FOV.
At the same time, changes in the spot intensity distribution
induced by defocus manifest as additional inflection points in
the response curves.
(a) received optical power levels = −18.6dBm
(b) received optical power levels = −55.0dBm
Fig. 4: QD output voltage ratios.
We then examine the conversion from voltage ratio to
estimated position. Figure 5 shows the estimated spot positions
px and estimation errors (px − xtrue) when applying the
conversion defined by Eq. 4 to the voltage ratio at d = 0.
For d̸ = 0, the estimated positions and errors obtained with
conversion coefficients K = 0.3, 0.35, π/8(≃ 0.392) are
shown in Fig. 6.
For the Airy distribution case (d = 0), Eq. 4 provided an
excellent match to the simulated response. Under defocused
conditions, however, the optimal coefficient depends on the
degree of defocus: for d = 2 mm, where the spot shape
remains close to an Airy distribution, K = 0.35 provided the
best agreement, while for d = 10 mm, where the spot shape
resembles a flat-top distribution, K = π/8 matched more
closely. These results demonstrate that the optimal conversion
coefficient varies with the amount of defocus.
Table I summarizes the conversion coefficients that min-
imize the root-mean-square (RMS) error of the estimated
(a) Estimated position (b) Estimation error
Fig. 5: Simulation results at d = 0 mm (The red dashed lines
indicate the linear FOV range).
(a) d = 2 mm
(b) d = 6 mm
(c) d = 10 mm
Fig. 6: Estimated positions and errors under defocused condi-
tions with various conversion coefficients K.
position within the linear FOV for several defocus values. The
results confirm that larger defocus leads to larger optimal K.
Although the linear FOV increases with defocus, the RMS
error within the linear region also increases, revealing a trade-
off between linear FOV and estimation accuracy. While the
table lists the coefficients that minimize RMS error, in practice,
if a tolerance for estimation error is specified, one can instead
select K to maximize the linear FOV while still satisfying the
error requirement.
TABLE I: Optimal conversion coefficient K, linear FOV, and
RMS error under different defocus values.
Defocus [mm] 2 4 6 8 10
Optimal K 0.343 0.390 0.395 0.398 0.406
Linear FOV [mm] 0.027 0.062 0.095 0.13 0.16
RMS [mm] 0.0027 0.0045 0.0046 0.0046 0.0048
IV. QD EXPERIMENT
A. Experimental Configuration
To validate the results obtained from the simulations in
the previous section, an experiment using an actual QD was
conducted. The experimental configuration is shown in Fig. 7.
In the setup, a 1550 nm laser beam with a 2.27 mm beam
diameter emitted from a fiber collimator was focused by a
front lens and then incident onto the QD, which was mounted
on a three-axis stage positioned near the focal plane.
Fig. 7: Experimental configuration of the QD setup.
First, with the defocus position fixed, the QD was translated
horizontally in steps of 0.01 mm, similar to the procedure em-
ployed in the simulation, and its output voltage corresponding
to the spot position was read out and recorded by Arduino.
By repeating this measurement while varying the defocus
distance, the variation of the QD response under different
defocus conditions was characterized.
The specifications of the QD (KPDE150Q-H15) and the
inverting amplifier used in the experiment are summarized in
Table II.
TABLE II: Specifications of the QD and inverting amplifier.
Quadrant Detector (QD)
Active area diameter 1.5 mm
Element gap 0.10 or 0.04 mm
Responsivity (@1550 nm) 1.0 A/W
I/V conversion resistance 200 kΩ
Dark current 0.3 nA
Inverting Amplifier
R1 22 kΩ
R2 39 kΩ
C 0.0001 μF
The focal length of the front lens for the QD was set
to 75 mm, consistent with the simulation described later.
The laser power was adjusted to 18.6 dBm using an optical
attenuator, such that the QD would just reach saturation when
all the laser power was incident on a single quadrant.
The entire experimental apparatus was placed on a
vibration-isolated optical table. During the measurements,
the surrounding environment was darkened by switching off
nearby electrical lighting and covering windows with blackout
curtains to minimize background light.