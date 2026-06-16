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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Analyzer module that loads \c SimEnergyDeposit and \c OpDetBacktrackerRecord
 * data from an \c art::Event and generates output data for comparison.
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

    // Initialize output file(s) and data objects
    void beginJob() override;

    // Read art::Event and generate output data
    void analyze(art::Event const& event) override;

  private:
    // Fcl input data
    art::InputTag sim_tag_;
    art::InputTag obtr_tag_;

    struct HistogramStore
    {
        using PDG = int;
        using PdgMap = std::map<PDG, TH1D*>;

        PdgMap step_len;
        TH1D* btr_time;
        TH2D* btr_detid_logtime_energy;
        TH2D* btr_detid_logtime_numphotons;
    } histograms_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
