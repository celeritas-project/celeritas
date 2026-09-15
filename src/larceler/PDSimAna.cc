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
#include <lardataobj/Simulation/SimPhotons.h>
#include <messagefacility/MessageLogger/MessageLogger.h>

#include "corecel/Assert.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with FHiCL input information.
 */
PDSimAna::PDSimAna(Parameters const& config)
    : art::EDAnalyzer{config}
    , sim_tag_{config().SimulationLabel()}
    , pd_tag_{config().PDModuleLabel()}
{
    inp_.num_channels = config().NumChannels();
    inp_.min_time = config().MinTime();
    inp_.max_time = config().MaxTime();
    CELER_VALIDATE(inp_.num_channels > 0,
                   << "invalid number of channels " << inp_.num_channels);
    CELER_VALIDATE(inp_.min_time < inp_.max_time,
                   << "min time is not greater than max time");
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
    constexpr double max_num_photons_per_edep{100000};
    hist_.edep_photons = tfs->make<TH1D>("edep_photons",
                                         "Energy deposits;# photons",
                                         100,
                                         0,
                                         max_num_photons_per_edep);

    hist_.hit_time
        = tfs->make<TH1D>("time", "Hit time/ns", 100, 0, inp_.max_time);

    auto time_grid = geomspace(inp_.min_time, inp_.max_time, 101);
    hist_.photons_detid_time
        = tfs->make<TH2D>("photons_detid_time",
                          "Photons;hit time/ns;opdet",
                          static_cast<int>(time_grid.size()) - 1,
                          time_grid.data(),
                          static_cast<int>(inp_.num_channels),
                          0.0,
                          static_cast<double>(inp_.num_channels));
}

//---------------------------------------------------------------------------//
/*!
 * Loop over event data and populate histograms.
 */
void PDSimAna::analyze(art::Event const& event)
{
    // Convert SimEnergyDeposits
    using VecSED = std::vector<sim::SimEnergyDeposit>;
    for (auto const& edep : *event.getValidHandle<VecSED>(sim_tag_))
    {
        this->fill(edep);
    }

    // Convert SimPhotonLites
    using VecSPL = std::vector<sim::SimPhotonsLite>;
    using VecBTR = std::vector<sim::OpDetBacktrackerRecord>;
    auto const& sim_photons = *event.getValidHandle<VecSPL>(pd_tag_);
    auto const& btrs = *event.getValidHandle<VecBTR>(pd_tag_);
    CELER_VALIDATE(sim_photons.size() >= btrs.size(),
                   << "expected sim photon size (" << sim_photons.size()
                   << ") to be same at least as large as btr size ("
                   << btrs.size() << ")");
    std::vector<bool> encountered(sim_photons.size());
    for (auto const& btr : btrs)
    {
        auto i = static_cast<std::size_t>(btr.OpDetNum());
        CELER_VALIDATE(i < encountered.size() && !encountered[i],
                       << "invalid or duplicate BTR OpDetNum " << i);
        CELER_ASSERT(i < sim_photons.size());
        this->fill(sim_photons[i], btr);
        encountered[i].flip();
    }
    auto num_missing
        = std::count(encountered.begin(), encountered.end(), false);
    if (num_missing > 0)
    {
        CELER_LOG(info) << "No hits encountered in " << num_missing
                        << " channels in event " << event.event();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Bin data from a SimEnergyDeposit.
 */
void PDSimAna::fill(sim::SimEnergyDeposit const& edep)
{
    // Step length by PDG
    int const pdg = edep.PdgCode();
    auto&& [iter, inserted] = hist_.pdg_edep_step_len.insert({pdg, nullptr});
    if (inserted)
    {
        std::string label = "step_len_" + std::to_string(pdg);
        std::string title = "Step length/cm: " + std::to_string(pdg);
        constexpr double min_len{0};
        constexpr double max_len{0.05};
        constexpr int num_bins{100};

        art::ServiceHandle<art::TFileService const> tfs;
        iter->second = tfs->make<TH1D>(
            label.c_str(), title.c_str(), num_bins, min_len, max_len);
    }
    iter->second->Fill(edep.StepLength());

    // Photons
    hist_.edep_photons->Fill(edep.NumPhotons());
}

//---------------------------------------------------------------------------//
/*!
 * Bin data from a SimPhotonsLite object (single channel).
 */
void PDSimAna::fill(sim::SimPhotonsLite const& spl,
                    sim::OpDetBacktrackerRecord const&)
{
    auto const opdet_id = static_cast<double>(spl.OpChannel);
    for (auto [tick, photons] : spl.DetectedPhotons)
    {
        hist_.photons_detid_time->Fill(tick, opdet_id, photons);
        hist_.hit_time->Fill(tick, photons);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
