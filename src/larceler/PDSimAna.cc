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
#include "corecel/cont/Range.hh"

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

    // Initialize histograms
    histograms_.sdp_zy
        = tfs->make<TH2D>("sdp_zy", "sdp_zy", 145, 0, 1450, 65, -650, 650);
    histograms_.btr_time = tfs->make<TH1D>(
        "obtr.timePDclock", "obtr.timePDclock", 100, 0, 15e3);
    histograms_.btr_detid_time = tfs->make<TH2D>(
        "btr_opdetid_time", "btr_opdetid_time", 500, 0, 500, 100, 0, 1.2e9);
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

    double total_time{0};
    for (auto const& btr : vec_obtr)
    {
        auto const opdet_id = btr.OpDetNum();
        for (auto const& map : btr.timePDclockSDPsMap())
        {
            auto const& time = map.first;
            auto const& vec_sdp = map.second;
            total_time += time;

            histograms_.btr_time->Fill(time);

            for (auto const& sdp : vec_sdp)
            {
                histograms_.sdp_zy->Fill(sdp.z, sdp.y);
            }
        }
        histograms_.btr_detid_time->Fill(opdet_id, total_time);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
