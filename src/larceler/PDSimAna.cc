//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/PDSimAna.cc
//---------------------------------------------------------------------------//
#include "PDSimAna.hh"

#include <cmath>
#include <art/Framework/Principal/Event.h>
#include <art/Framework/Principal/Handle.h>
#include <art/Framework/Principal/Run.h>
#include <art/Framework/Principal/SubRun.h>
#include <art_root_io/TFileService.h>
#include <canvas/Utilities/InputTag.h>
#include <fhiclcpp/ParameterSet.h>
#include <larcore/CoreUtils/ServiceUtil.h>
#include <larcorealg/Geometry/OpDetGeo.h>
#include <lardataobj/Simulation/OpDetBacktrackerRecord.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>
#include <messagefacility/MessageLogger/MessageLogger.h>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/grid/VectorUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with FHiCL input information.
 */
PDSimAna::PDSimAna(Parameters const& config)
    : art::EDAnalyzer{config}
    , sim_tag_{config().SimulationLabel()}
    , obtr_tag_{config().ModuleLabel()}
{
}

//---------------------------------------------------------------------------//
/*!
 * Create ROOT file and initialize histograms.
 */
void PDSimAna::beginJob()
{
    // ROOT file creation and writing is managed by the TFileService
    art::ServiceHandle<art::TFileService const> tfs;

    // Create log bins
    size_t nbins_time = 100;
    double min_time = 10;  // [ns]
    double max_time = 20e3;  // [ns]
    auto vec_log_bins = geomspace(min_time, max_time, nbins_time + 1);

    // Initialize histograms
    histograms_.btr_time
        = tfs->make<TH1D>("btr_timepdclock", "btr_timepdclock", 100, 0, 50e3);
    histograms_.btr_detid_logtime_energy
        = tfs->make<TH2D>("btr_logtime_opdetid_energy",
                          "btr_logtime_opdetid_energy",
                          nbins_time,
                          vec_log_bins.data(),
                          480,
                          0,
                          480);
    histograms_.btr_detid_logtime_numphotons
        = tfs->make<TH2D>("btr_logtime_opdetid_numphotons",
                          "btr_logtime_opdetid_numphotons",
                          nbins_time,
                          vec_log_bins.data(),
                          480,
                          0,
                          480);
}

//---------------------------------------------------------------------------//
/*!
 * Loop over event data and populate histograms.
 */
void PDSimAna::analyze(art::Event const& event)
{
    using VecSimEdep = std::vector<sim::SimEnergyDeposit>;
    using VecOpDetBTR = std::vector<sim::OpDetBacktrackerRecord>;

    // Load SimEnergyDeposit and OpDetBacktrackerRecord data
    auto const& vec_simedep = *(event.getValidHandle<VecSimEdep>(sim_tag_));
    auto const& vec_obtr = *(event.getValidHandle<VecOpDetBTR>(obtr_tag_));

    // Initialize step length histogram for a given PDG encoding
    auto init_step_len_hist = [](int pdg) {
        art::ServiceHandle<art::TFileService const> tfs;
        return tfs->make<TH1D>(
            Form("step_len_%d", pdg), Form("step_len_%d", pdg), 100, 0, 0.05);
    };

    // Step length from SimEnergyDeposit, mapped by PDG
    for (auto const& edep : vec_simedep)
    {
        auto const pdg = edep.PdgCode();
        if (histograms_.step_len.find(pdg) == histograms_.step_len.end())
        {
            histograms_.step_len[pdg] = init_step_len_hist(pdg);
        }
        histograms_.step_len[pdg]->Fill(edep.StepLength());  // [cm]
    }

    // OpDetBacktrackerRecord data plots
    for (auto const& btr : vec_obtr)
    {
        auto const opdet_id = btr.OpDetNum();
        for (auto const& map : btr.timePDclockSDPsMap())
        {
            auto const& time = map.first;  // [ns]
            auto const& vec_sdp = map.second;  // vector of SDP objects

            double total_energy_per_timeclock{0};
            double total_numphotons_per_timeclock{0};
            for (auto const& sdp : vec_sdp)
            {
                total_energy_per_timeclock += sdp.energy;  // [MeV]
                total_numphotons_per_timeclock += sdp.numPhotons;
            }

            histograms_.btr_time->Fill(time);
            histograms_.btr_detid_logtime_energy->Fill(
                time, opdet_id, total_energy_per_timeclock);
            histograms_.btr_detid_logtime_numphotons->Fill(
                time, opdet_id, total_numphotons_per_timeclock);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
