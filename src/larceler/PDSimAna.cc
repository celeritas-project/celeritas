//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/PDSimAna.cc
//---------------------------------------------------------------------------//
#include "PDSimAna.hh"

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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with FHiCL input information.
 */
PDSimAna::PDSimAna(Parameters const& config)
    : art::EDAnalyzer{config}
    , sim_tag_{config().SimulationLabel()}
    , btr_tag_{config().ModuleLabel()}
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

    // Initialize histograms
    histograms_.sdp_energy
        = tfs->make<TH1D>("sdp_energy", "sdp_energy", 100, 0, 0.01);
    histograms_.pd_time = tfs->make<TH1D>("pd_time", "pd_time", 100, 0, 15e3);
    histograms_.opdet_energy = tfs->make<TH2D>(
        "opdet_energy", "opdet_energy", 500, 0, 500, 100, 0, 0.01);
    histograms_.btr_time_energy = tfs->make<TH2D>(
        "btr_time_energy", "btr_time_energy", 100, 0, 15e3, 100, 0, 0.01);
}

//---------------------------------------------------------------------------//
/*!
 * Loop over event data and populate histograms.
 */
void PDSimAna::analyze(art::Event const& e)
{
    // Load SimEnergyDeposit and OpDetBacktrackerRecord data
    auto const& sim_edeps
        = *(e.getValidHandle<std::vector<sim::SimEnergyDeposit>>(sim_tag_));
    auto const& opdet_btrs = *(
        e.getValidHandle<std::vector<sim::OpDetBacktrackerRecord>>(btr_tag_));

    for (auto const& btr : opdet_btrs)
    {
        auto opdet_id = btr.OpDetNum();
        double total_btr_energy{0};
        for (auto const& map : btr.timePDclockSDPsMap())
        {
            // histograms_.hist->Fill(sdp.second.energy());
            auto const& time = map.first;
            auto const& vec_sdp = map.second;

            histograms_.pd_time->Fill(time);
            double total_sdp_energy{0};
            for (auto const& sdp : vec_sdp)
            {
                histograms_.sdp_energy->Fill(sdp.energy);
                total_btr_energy += sdp.energy;
                total_sdp_energy += sdp.energy;
            }
            histograms_.btr_time_energy->Fill(time, total_sdp_energy);
        }
        histograms_.opdet_energy->Fill(opdet_id, total_btr_energy);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
