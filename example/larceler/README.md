LArSoft workflow example
========================

# Overview

This set of `fcl` job files allow executing neutrino event generation in GENIE,
energy deposits in Geant4, propagation of optical photons (fast and full), and
producing analysis data for comparison. The diagram below shows the workflow
with the respective `fcl` files:

```mermaid
flowchart LR
evt-gen["GENIE
    prodgenie_nu_dune10kt_1x2x6.fcl"]
genie-data[("art::Event
    EventGen")]
det-edep["LArG4 + IonAndScint
    dune10kt_larg4_1x2x6.fcl"]
edep-data[("art::Event
    SimEnergyDeposit
    (label: IonAndScint)")]
det-optical-fast["Optical fast simulation
    dune10kt_opticalsim_1x2x6.fcl"]
det-optical-full["Celeritas optical transport
    dune10kt_opticalsim_celeritas_1x2x6.fcl"]
optical-data[("art::Event
    OpDetBackTrackerRecord
    (label: opticalsim)")]
analyzer["PDSimAna (analyzer)
    pdsimana_run.fcl"]
analyzer-data[("analysis files")]

evt-gen --> genie-data --> det-edep --> edep-data
edep-data --> det-optical-fast & det-optical-full --> optical-data --> analyzer
analyzer --> analyzer-data
```

The next sections briefly describe each component and how to invoke them,
assuming `dunesw` _and_ Celeritas are correctly set up.


# Generating GENIE samples

- Use default
  [prodgenie_nu_dune10kt.fcl](https://internal.dunescience.org/doxygen/prodgenie__nu__dune10kt__1x2x6_8fcl_source.html),
  which in your `PATH` through `dunesw`.
```sh
$ lar -c prodgenie_nu_dune10kt_1x2x6.fcl [optional: -n num_events] -o genie-output.root
```
- If `-n` is not used, the default number of events is set to 10 (defined
  upstream in
  [prodgenie_common_dunefd.fcl](https://internal.dunescience.org/doxygen/prodgenie__common__dunefd_8fcl_source.html))


# Running LArG4 + IonAndScint

- Use local `dune10k_larg4_1x2x6.fcl` file
- To loop over a subset of events, replace `-s` by `-n [num_events]`
```sh
$ lar -c dune10k_larg4_1x2x6.fcl -s genie-output.root -o larg4-output.root
```


# Running optical simulations

- Use local `dune10k_optical_1x2x6.fcl` and
  `dune10k_optical_celeritas_1x2x6.fcl`  files
- To loop over a subset of events, replace `-s` by `-n [num_events]`

## Fast simulation
```sh
$ lar -c dune10k_optical_1x2x6.fcl -s larg4-output.root -o fastsim-output.root
```

## Celeritas
```sh
$ lar -c dune10k_optical_celeritas_1x2x6.fcl -s larg4-output.root -o celeritas-output.root
```


# Analyzing the optical simulation data

- Use local `pdsimana_run.fcl` file
- To loop over a subset of events, replace `-s` by `-n [num_events]`
- As noted in the `PDSimAna.fcl` documentation, the `ModuleLabel: "opticalsim"`
  in `PDSimAna.fcl` is correct if optical simulation is generated with the local
  `*optical*.fcl` files. If fast simulation is generated from a default LArSoft
  `fcl` job, that will likely be `PDFastSim`
```sh
$ lar -c pdsimana_run.fcl -s fastsim-output.root -o fastsim-ana-output.root
$ lar -c pdsimana_run.fcl -s celeritas-output.root -o celeritas-ana-output.root
```
