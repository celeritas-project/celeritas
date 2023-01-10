.. Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
.. See the doc/COPYRIGHT file for details.
.. SPDX-License-Identifier: CC-BY-4.0

.. _release_history:

***************
Release History
***************

Version 0.2.0
=============

Version 0.2.0 enables new coupled integration between Celeritas and Geant4
using the Acceleritas interface library. It features new ROOT output capabilities
including MC truth output, a set of high-level API classes for integrating
into existing Geant4 toolkit-based apps, and better support for multithreaded
use.

New features
------------

* Save JSON exception output if demo loop fails *(@sethrj, #507)*
* Export SB, Livermore PE, and atomic relaxation data to ROOT file *(@stognini, #481)*
* Refactor ORANGE input to be more extensible *(@sethrj, #510)*
* Make primary generator more general *(@amandalund, #514)*
* Support arbitrary user/internal kernels as part of stepping loop *(@sethrj, #525)*
* Improve CMake version/host metadata *(@sethrj, #529)*
* Baby steps toward Geant4 integration *(@sethrj, #531)*
* Add "background" volume support to ORANGE *(@sethrj, #530)*
* Build Livermore/SB data from ImportedData *(@sethrj, #535)*
* Load and build multiple universes in ORANGE *(@elliottbiondo, #534)*
* Support adding primaries at any step in the transport loop *(@amandalund, #542)*
* Add basic step collector *(@sethrj, #544)*
* Add finer granularity to step collector *(@sethrj, #549)*
* Add a Celeritas handler for ROOT Error (messages) *(@pcanal, #552)*
* Enable resizing of CUDA malloc heap to fix VecGeom failures *(@mrguilima, #554)*
* Add detector mapping and filtering to hit collector *(@sethrj, #555)*
* Add helper class for extracting detector hits to CPU *(@sethrj, #559)*
* Add initialization of particles in multi-universe ORANGE geometries *(@elliottbiondo, #546)*
* Add upper_bound functions to corecel/math *(@elliottbiondo, #565)*
* Add ROOT MC truth output *(@stognini, #564)*
* Handle exceptions from inside OpenMP parallel pragmas *(@sethrj, #563)*
* Add skeleton classes for Celeritas/Geant4 integration *(@sethrj, #567)*
* Add thread-local transporter and Celeritas shared params setup to Acceleritas *(@amandalund, #575)*
* Set initial values of SetupOptions parameters from GlobalSetup *(@mrguilima, #576)*
* Add Geant4 Exception converter *(@sethrj, #580)*
* Convert Thrust exceptions to Geant4 *(@sethrj, #582)*
* Add diagnostic output to geant demo *(@sethrj, #583)*
* Auto-export GDML from Geant4 geometry *(@sethrj, #585)*
* Support user-defined along-step kernels in accel+demo *(@sethrj, #586)*
* Add hit processor to convert Celeritas detector hits to Geant4 *(@sethrj, #581)*
* Add HepMC3 reader to `demo-geant-integration` *(@stognini, #578)*
* Add track parent id information to step data *(@stognini, #590)*
* Enable all physics in geant demo for true offloading *(@sethrj, #591)*

Reviewers: @sethrj *(12)*, @amandalund *(11)*, @stognini *(2)*, @paulromano *(2)*, @whokion *(2)*, @tmdelellis *(1)*, @pcanal *(1)*, @elliottbiondo *(1)*, @mrguilima *(1)*

Bug fixes
---------

* Fix infrastructure and build issues for summit *(@sethrj, #509)*
* Fix class name conflict, remove default initializers, and add other tweaks *(@sethrj, #504)*
* Fix indexing of imported micro xs *(@amandalund, #512)*
* Fix JSON build issues *(@sethrj, #536)*
* Fix library location and configure issues from #526 *(@sethrj, #537)*
* Fix thread safety issues in Celeritas *(@sethrj, #532)*
* Do not include ROOT's CMake "use" file to avoid potential nlohmann_json conflicts *(@drbenmorgan, #556)*
* Remove calls to host code from host-device Collection *(@sethrj, #547)*
* Fix celeritas-gen when run from root directory *(@sethrj, #562)*
* Fix and work around some issues on HIP/Crusher *(@sethrj, #558)*
* Fix build documentation and issues with newer toolchains *(@sethrj, #571)*
* Minor fixes for older Geant4/VecGeom releases *(@sethrj, #587)*
* Use Geant4 environment variables to choose run manager and threads *(@sethrj, #589)*

Reviewers: @amandalund *(9)*, @sethrj *(7)*, @pcanal *(5)*, @whokion *(3)*, @elliottbiondo *(1)*, @tmdelellis *(1)*, @paulromano *(1)*

Documentation improvements
--------------------------

* Add granularity to Geant setup *(@sethrj, #485)*
* Format code base (clang-format version 14.0.6) *(@sethrj, #506)*
* Use `test` sub-namespace instead of `celeritas_test` *(@sethrj, #503)*
* Refactor ORANGE data and simple tracker to support nested universes *(@sethrj, #520)*
* Define gauss as internal field strength and use tesla for input *(@sethrj, #522)*
* Break library into multiple parts and mirror install tree *(@sethrj, #526)*
* Add include-what-you-use script and pragmas *(@sethrj, #540)*
* Split orange/Types and add Orange prefix to Data *(@sethrj, #541)*
* Update CUDA RDC CMake code *(@pcanal, #545)*
* Fix git-clang-format hook and code documentation *(@sethrj, #568)*
* Change Transport/Stepper interfaces to take span of `Primary` instead of vector *(@paulromano, #572)*
* Refactor geant demo app and `accel` code *(@sethrj, #577)*
* Move HepMC3 reader to accel and make a little more reusable *(@sethrj, #593)*
* Rename `CELER_TRY_ELSE` to `CELER_TRY_HANDLE` *(@sethrj, #594)*
* Add Acceleritas documentation *(@sethrj, #595)*
* Update IWYU and apply to codebase *(@sethrj, #596)*
* Update clang-format to avoid alignment and use "east const" *(@sethrj, #574)*
* Update copyrights for 2023 *(@sethrj, #598)*

Reviewers: @amandalund *(9)*, @stognini *(2)*, @sethrj *(2)*, @paulromano *(1)*, @elliottbiondo *(1)*

Minor internal changes
----------------------

* Add release procedure, roles, and 0.1.x release notes *(@sethrj, #519)*
* Add DOE DOI and improve PR process documentation *(@sethrj, #533)*
* Add a simple SensitiveHit for demo-geant4-integration *(@whokion, #579)*

Reviewers: @paulromano *(2)*, @tmdelellis *(1)*, @pcanal *(1)*, @sethrj *(1)*, @whokion *(1)*

Version 0.1.4
=============

This version fixes significant errors in along-step tracking when multiple
scattering and/or magnetic fields are in use.

Bug fixes
---------

* Fix additional failures resulting from direction change on a boundary after crossing *(@amandalund, #517)*
* Fix the true path length limit of UrbanMsc *(@whokion, #521)*
* Fix field propagation for stuck and long-track particles *(@sethrj, #518)*
* Don't slow particles to zero when hitting boundary *(@sethrj, #524)*
* Cache multiple scattering range at the first step in a new volume *(@whokion, #527)*
* Reset physics state when a secondary is initialized in the parent's track slot *(@amandalund, #528)*

Version 0.1.3
=============

This patch fixes additional issues with tracking in a magnetic field.

Bug fixes
---------

* Fix near-infinite loop in field propagation *(@sethrj, #511)*
* Allow tracks taking small steps to propagate in field *(@amandalund, #513)*
* Fix ability to disable Rayleigh scattering when running demo-loop app
* Fix failure when changing direction on boundary immediately after crossing *(@sethrj, #515)*
* Escape rather than segfault on boundary crossing failure in release mode in ORANGE *(@sethrj, #516)*

Version 0.1.2
=============

This is a bug-fixing patch that addresses a few outstanding build issues,
targeted at making it easier to run and debug the regression problem suite.

New features
------------

* Save JSON exception output if demo loop fails by *(@sethrj, #507)*

Bug fixes
---------

* Fix class name conflict, remove default initializers, and tweak field driver *(@sethrj, #504)*
* Fix no-ROOT demo loop for 0.1.x and other summit changes by *(@sethrj, #508)*

Version 0.1.1
=============

This is a bug-fixing patch that addresses numerous outstanding issues with the
transport loop.

New features
------------

* Filter imported data from Geant4 based on user options by *(@sethrj, #482)*
* Add contributing guide and development docs by *(@sethrj, #502)*

Bug fixes
---------

* Fix uninitialized data access in primary generator by *(@sethrj, #472)*
* Build processes based on exported data by *(@sethrj, #483)*
* NVHPC: silence warnings by *(@sethrj, #486)*
* Bring latest upstream updates to the BVHNavigator by *(@mrguilima, #484)*
* Force disabling of shared libs for VecGeom 1.2.0+ by *(@sethrj, #489)*
* Add minor fixes for UrbanMsc by *(@whokion, #492)*
* Set remaining model lower limits to zero and make sure demo loop runs with integral approach off by *(@amandalund, #490)*
* Fix VecGeomTrack::move_internal assertion failure by *(@mrguilima, #493)*
* Fix clang-14/ROCM build with JSON enabled by *(@sethrj, #496)*
* Better workaround for VecGeom 1.2 crashes by *(@sethrj, #495)*
* Fix imported model microscopic cross sections by *(@amandalund, #487)*
* Fix unintentional MSC enable and some displacement logic by *(@sethrj, #500)*
* Fix unusual errors in field propagation by *(@sethrj, #499)*
* Fix vecgeom boundary state and add more thorough testing by *(@sethrj, #494)*
* Improve static/shared CUDA library choice when linking VecGeom by *(@pcanal, #497)*
* Fix ORANGE reentrant boundary crossings by *(@sethrj, #501)*

Internal changes
----------------

* Rewrite field tests by *(@sethrj, #471)*
* Add range-to-step tests and remove min by *(@sethrj, #479)*
* Add unit tests and make minor changes to FieldDriver by *(@sethrj, #478)*
* Unify units in EM interactors/data by *(@sethrj, #477)*
* Add tests for field propagator near boundaries by *(@sethrj, #480)*
* Return one event at a time from EventReader and PrimaryGenerator by *(@amandalund, #488)*
* Infrastructure changes for geo heuristic test by *(@sethrj, #498)*

Version 0.1.0
=============

"Initial viable product" release of Celeritas. This release is focused on unit
testing and stabilizing the API for exploratory use by downstream applications
and frameworks. Some initial code verification problems (looking at energy and
step distributions for simple problems such as TestEM3) are promising, but the
code is far from validated.

New features
------------

- Standard EM physics including multiple scattering and energy loss
  fluctuations
- VecGeom for GDML navigation, ORANGE for testing
- Support for CUDA, HIP, and CPU (with and without OpenMP)
- Magnetic field support with runtime-swappable field types

Known bugs
----------

- Magnetic field propagation can fail at geometry boundaries, especially with
  ORANGE.
- Multiple scattering requires more steps than expected compared to Geant4.

Contributors
------------

Thanks to the contributors at Argonne National Lab, Lawrence Berkeley Lab,
Fermilab, Oak Ridge National Laboratory, and other collaborating institutions.

- Philippe Canal (@pcanal): #115, #119, #130, #171, #172, #166, #222, #239,
  #242, #243, #254, #338
- Doaa Deeb (@DoaaDeeb): #257
- Tom Evans (@tmdelellis): #57, #65, #112
- Seth R Johnson (@sethrj): #4, #8, #13, #16, #15, #21, #24, #25, #32, #17,
  #35, #39, #36, #40, #45, #44, #50, #51, #61, #59, #62, #63, #64, #69, #78,
  #80, #79, #76, #73, #82, #83, #84, #85, #86, #87, #88, #91, #92, #93, #95,
  #99, #98, #106, #104, #108, #114, #116, #105, #117, #118, #122, #120, #124,
  #128, #129, #133, #135, #131, #136, #137, #138, #139, #142, #143, #144, #141,
  #147, #148, #151, #149, #153, #150, #156, #157, #162, #160, #170, #168, #174,
  #169, #177, #178, #179, #184, #189, #190, #193, #195, #191, #199, #204, #196,
  #205, #201, #207, #208, #211, #218, #217, #219, #220, #224, #225, #226, #228,
  #235, #237, #236, #238, #247, #276, #292, #293, #294, #296, #298, #291, #306,
  #301, #307, #311, #313, #314, #315, #312, #321, #322, #325, #329, #331, #332,
  #335, #333, #336, #337, #340, #339, #343, #344, #342, #345, #347, #348, #349,
  #351, #359, #360, #364, #365, #366, #378, #374, #373, #379, #381, #384, #380,
  #387, #386, #388, #389, #391, #394, #393, #395, #397, #398, #399, #400, #405,
  #403, #404, #410, #408, #406, #411, #402, #414, #415, #417, #416, #422, #424,
  #426, #427, #428, #433, #432, #435, #436, #434, #437, #441, #439, #445, #443,
  #448, #449, #453, #456, #455, #458, #457, #464, #465, #468
- Soon Yung Jun (@whokion): #41, #70, #173, #200, #214, #221, #230, #250, #259,
  #258, #260, #316, #317, #320, #324, #370, #375, #390, #396, #407, #413, #430,
  #454, #467
- Guilherme Lima (@mrguilima): #42, #38, #109, #90, #167, #229, #234, #232,
  #328, #383, #446, #452
- Amanda Lund (@amandalund): #6, #20, #47, #52, #89, #100, #113, #134, #154,
  #159, #161, #186, #185, #198, #216, #215, #209, #227, #240, #245, #255, #251,
  #264, #274, #269, #285, #290, #297, #304, #309, #319, #323, #330, #346, #350,
  #353, #362, #368, #369, #372, #376, #382, #385, #401, #440, #444, #450, #463
- Ben Morgan (@drbenmorgan): #53, #56, #110, #121, #367, #371
- Vincent R Pascuzzi (@vrpascuzzi): #68, #72, #111, #241, #248, #246, #287
- Paul Romano (@paulromano): #107, #197, #265, #268, #270, #275, #273, #289,
  #299, #303, #305, #308, #310, #318
- Stefano C Tognini (@stognini): #30, #55, #81, #132, #175, #188, #194, #203,
  #210, #231, #244, #271, #302, #327, #326, #341, #423
