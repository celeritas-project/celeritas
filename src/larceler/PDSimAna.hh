//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/PDSimAna.hh
//---------------------------------------------------------------------------//
#pragma once

#include <TH1D.h>
#include <TH2D.h>
#include <art/Framework/Core/EDAnalyzer.h>
#include <canvas/Utilities/InputTag.h>
#include <fhiclcpp/types/Atom.h>
#include <fhiclcpp/types/Sequence.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Analyzer module that exports detector geometry information and, \em
 * optionally , \c sim::SimEnergyDeposit data to basic ROOT types for use in
 * - Celeritas unit tests (e.g. LarStandaloneRunner); or
 * - Non-LarSoft applications (e.g. a Geant4 offloading app).
 *
 * The simplified TTree does not require dictionaries. Each TTree entry is one
 * event. Each TBranch is a vector of \c sim::SimEnergyDeposit data, and thus
 * each TBranch entry is associated to a \c sim::SimEnergyDeposit object.
 *
 * Usage:
 * Export detector geometry data only:
 * $ lar -c job.fcl
 * Export detector geometry \em and simulation data:
 * $ lar -c job.fcl -s [geant4-output.root]
 *
 * To store only a subset of events, use the optional `-n [num_events]` flag.
 */
class PDSimAna : public art::EDAnalyzer
{
  public:
    struct Config
    {
        fhicl::Atom<art::InputTag> SimulationLabel{
            fhicl::Name("SimulationLabel"),
            fhicl::Comment(R"(SimEnergyDeposit event tag)")};

        fhicl::Atom<art::InputTag> ModuleLabel{
            fhicl::Name{"ModuleLabel"},
            fhicl::Comment{R"(OpdetBacktrackerRecord event tag)"}};
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

    // Create histograms with sim energy deposit data
    void beginJob() override;

    // Export simulation data from input file
    void analyze(art::Event const& event) override;

  private:
    // Fcl input data
    art::InputTag sim_tag_;
    art::InputTag btr_tag_;

    struct HistogramStore
    {
        TH1D* pd_time;
        TH1D* sdp_energy;
        TH2D* opdet_energy;
        TH2D* btr_time_energy;
    } histograms_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
