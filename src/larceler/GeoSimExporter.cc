//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/GeoSimExporter.cc
//---------------------------------------------------------------------------//
#include "GeoSimExporter.hh"

#include <art/Framework/Principal/Event.h>
#include <art/Framework/Principal/Handle.h>
#include <art/Framework/Principal/Run.h>
#include <art/Framework/Principal/SubRun.h>
#include <art_root_io/TFileService.h>
#include <canvas/Utilities/InputTag.h>
#include <fhiclcpp/ParameterSet.h>
#include <larcore/CoreUtils/ServiceUtil.h>
#include <larcore/Geometry/Geometry.h>
#include <larcorealg/Geometry/OpDetGeo.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>
#include <messagefacility/MessageLogger/MessageLogger.h>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with GDML geometry and export its information.
 */
GeoSimExporter::GeoSimExporter(Parameters const& config)
    : art::EDAnalyzer{config}
    , sim_tag_{config().SimulationLabel()}
    , max_edeps_(config().MaxEdepsPerEvent())
{
    // TTree and ROOT file writing is done automatically by the TFileService
    art::ServiceHandle<art::TFileService const> tfs;

    // Geometry information
    auto* det_info = tfs->make<TTree>("detector_info", "detector_info");

    auto const* geo = lar::providerFrom<geo::Geometry>();
    CELER_ASSERT(geo);
    std::string name = geo->DetectorName();

    det_info->Branch("name", &name);
    det_info->Fill();

    auto* geo_data = tfs->make<TTree>("optical_detectors", "optical_detectors");
    std::array<double, 3> pos;
    std::string info;
    geo_data->Branch("pos", &pos);
    geo_data->Branch("info", &info);

    for (unsigned int i = 0; i < geo->NOpDets(); i++)
    {
        auto const& opdet = geo->OpDetGeoFromOpDet(i);
        auto const& center = opdet.GetCenter();

        info = opdet.OpDetInfo(/* indent = */ "", /* verbosity = */ 1);
        pos = {center.x(), center.y(), center.z()};
        geo_data->Fill();
    }

    mf::LogInfo("GeoSimExporterModule") << "Saved detector information to "
                                           "root file";
}

//---------------------------------------------------------------------------//
/*!
 * Create TTree with sim data.
 */
void GeoSimExporter::beginJob()
{
    // TTree and ROOT file writing is done automatically by the TFileService
    art::ServiceHandle<art::TFileService const> tfs;

    sim_tree_ = tfs->make<TTree>("sim_energy_deposits", "sim_energy_deposits");

#define GSDE_CREATE_SIM_BRANCH(MEMBER) \
    sim_tree_->Branch(#MEMBER, &sim_edep_data_.MEMBER);

    // Branch names mimic sim::SimEnergyDeposit class getters
    GSDE_CREATE_SIM_BRANCH(NumPhotons);
    GSDE_CREATE_SIM_BRANCH(NumElectrons);
    GSDE_CREATE_SIM_BRANCH(ScintYieldRatio);
    GSDE_CREATE_SIM_BRANCH(Energy);
    GSDE_CREATE_SIM_BRANCH(Time);
    GSDE_CREATE_SIM_BRANCH(StartX);
    GSDE_CREATE_SIM_BRANCH(StartY);
    GSDE_CREATE_SIM_BRANCH(StartZ);
    GSDE_CREATE_SIM_BRANCH(EndX);
    GSDE_CREATE_SIM_BRANCH(EndY);
    GSDE_CREATE_SIM_BRANCH(EndZ);
    GSDE_CREATE_SIM_BRANCH(StartT);
    GSDE_CREATE_SIM_BRANCH(EndT);
    GSDE_CREATE_SIM_BRANCH(TrackID);
    GSDE_CREATE_SIM_BRANCH(PdgCode);

#undef GSDE_CREATE_SIM_BRANCH
}

//---------------------------------------------------------------------------//
/*!
 * Loop over optional larg4 Geant4 output simulation file event data with
 * \c IonAndScint objects and export test data.
 */
void GeoSimExporter::analyze(art::Event const& e)
{
    art::Handle<std::vector<sim::SimEnergyDeposit>> energy_deps;
    if (!e.getByLabel(sim_tag_, energy_deps))
    {
        mf::LogError("GeoSimExporterModule")
            << "Cannot find IonAndScint label. Either 1) missing input file "
               "(lar -c thisjob.fcl -s [geant4_output.root]) or "
               "2) missing IonAndScint data in art::Event";
        return;
    }

    // Verify if data is present
    int const edeps_size = (*energy_deps).size();
    if (edeps_size == 0)
    {
        mf::LogWarning("GeoSimExporterModule")
            << "sim::SimEnergyDeposit data is valid but has zero entries; "
               "Skipping event";
        return;
    }

    // Clear all vectors before pushing back event data
    this->clear();

    // If the requested maximum number of energy deposits per event is <= 0,
    // store all. Otherwise, set the limit to be up to the size of the vector
    // to avoid a segfault
    int const num_edeps_stored = (max_edeps_ <= 0)           ? edeps_size
                                 : (max_edeps_ > edeps_size) ? edeps_size
                                                             : max_edeps_;

#define GSDE_GET(MEMBER) sim_edep_data_.MEMBER->push_back(edep.MEMBER());

    for (int i = 0; i < num_edeps_stored; i++)
    {
        auto const& edep = (*energy_deps)[i];

        GSDE_GET(NumPhotons);
        GSDE_GET(NumElectrons);
        GSDE_GET(ScintYieldRatio);
        GSDE_GET(Energy);
        GSDE_GET(Time);
        GSDE_GET(StartX);
        GSDE_GET(StartY);
        GSDE_GET(StartZ);
        GSDE_GET(EndX);
        GSDE_GET(EndY);
        GSDE_GET(EndZ);
        GSDE_GET(StartT);
        GSDE_GET(EndT);
        GSDE_GET(TrackID);
        GSDE_GET(PdgCode);
    }

    sim_tree_->Fill();

    mf::LogInfo("GeoSimExporterModule")
        << "Wrote " << num_edeps_stored
        << " SimEnergyDeposition object(s) to ROOT file";

#undef GSDE_GET
}

//---------------------------------------------------------------------------//
/*!
 * Clear all \c sim::SimEnergyDeposit vector data before an event.
 */
void GeoSimExporter::clear()
{
#define GSDE_CLEAR(MEMBER) sim_edep_data_.MEMBER->clear();

    GSDE_CLEAR(NumPhotons);
    GSDE_CLEAR(NumElectrons);
    GSDE_CLEAR(ScintYieldRatio);
    GSDE_CLEAR(Energy);
    GSDE_CLEAR(Time);
    GSDE_CLEAR(StartX);
    GSDE_CLEAR(StartY);
    GSDE_CLEAR(StartZ);
    GSDE_CLEAR(EndX);
    GSDE_CLEAR(EndY);
    GSDE_CLEAR(EndZ);
    GSDE_CLEAR(StartT);
    GSDE_CLEAR(EndT);
    GSDE_CLEAR(TrackID);
    GSDE_CLEAR(PdgCode);

#undef GSDE_CLEAR
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
