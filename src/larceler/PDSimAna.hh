//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/PDSimAna.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <TH1D.h>
#include <TH2D.h>
#include <art/Framework/Core/EDAnalyzer.h>
#include <canvas/Utilities/InputTag.h>
#include <fhiclcpp/types/Atom.h>
#include <fhiclcpp/types/Sequence.h>

namespace sim
{
class SimEnergyDeposit;
class OpDetBacktrackerRecord;
class SimPhotonsLite;
}  // namespace sim

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Analyzer module that loads \c SimEnergyDeposit and \c OpDetBacktrackerRecord
 * data from an \c art::Event and generates output data for comparison.
 *
 * Related:
 * -
 * https://github.com/DUNE/duneopdet/blob/develop/duneopdet/OpticalDetector/Ana/OpDetAnalyzer.fcl
 */
class PDSimAna : public art::EDAnalyzer
{
  public:
    struct Config
    {
        fhicl::Atom<art::InputTag> SimulationLabel{
            fhicl::Name("SimulationLabel"),
            fhicl::Comment(R"(Module label containing SimEnergyDeposit)")};

        fhicl::Atom<art::InputTag> PDModuleLabel{
            fhicl::Name{"PDModuleLabel"},
            fhicl::Comment{R"(Module containing SimPhotonsLite/OpDetBTRs)"}};

        fhicl::Atom<unsigned int> NumChannels{
            fhicl::Name{"NumChannels"},
            fhicl::Comment{R"(Number of detector channels)"},
            480};

        fhicl::Atom<double> MinTime{
            fhicl::Name{"MinTime"},
            fhicl::Comment{R"(Minimum time for histogram binning [ns])"},
            10.0};

        fhicl::Atom<double> MaxTime{
            fhicl::Name{"MaxTime"},
            fhicl::Comment{R"(Maximum time for histogram binning [ns])"},
            20e3};
    };
    using Parameters = art::EDAnalyzer::Table<Config>;

  public:
    // Construct with input parameters and export geometry data
    explicit PDSimAna(Parameters const& p);

    //!@{
    // Prevent copy and assignment operations
    PDSimAna(PDSimAna const&) = delete;
    PDSimAna(PDSimAna&&) = delete;
    PDSimAna& operator=(PDSimAna const&) = delete;
    PDSimAna& operator=(PDSimAna&&) = delete;
    //!@}

    // Initialize output file(s) and data objects
    void beginJob() override;

    // Read art::Event and generate output data
    void analyze(art::Event const& event) override;

  private:
    struct Dims
    {
        unsigned int num_channels{0};
        double min_time{1e100};
        double max_time{0};
    };

    struct Histograms
    {
        using PDG = int;

        TH1D* edep_photons{nullptr};
        TH1D* hit_time{nullptr};
        std::map<PDG, TH1D*> pdg_edep_step_len;
        TH2D* photons_detid_time{nullptr};
    };

    // Fcl input data
    art::InputTag sim_tag_;
    art::InputTag pd_tag_;
    Dims inp_;

    // Mutable data
    Histograms hist_;
    // TODO: `Dims encountered_` for error reporting

    void fill(sim::SimEnergyDeposit const& edep);
    void fill(sim::SimPhotonsLite const& spl,
              sim::OpDetBacktrackerRecord const& btr);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
