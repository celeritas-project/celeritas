//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarDataReader.cc
//---------------------------------------------------------------------------//
#include "LarDataReader.hh"

#include <TDirectory.h>
#include <TFile.h>
#include <TTree.h>
#include <lardataobj/Simulation/SimEnergyDeposit.h>

#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with ROOT filename.
 */
LarDataReader::LarDataReader(std::string name)
    : root_file_(TFile::Open(name.c_str(), "read"))
{
    CELER_EXPECT(root_file_);
    CELER_EXPECT(!root_file_->IsZombie());

    data_dir_.reset(root_file_->GetDirectory(this->data_dir_name()));
    CELER_EXPECT(data_dir_);

    sim_tree_ = data_dir_->Get<TTree>(this->sim_data_tree_name());
    CELER_ASSERT(sim_tree_);

#define LDR_SET_SIM_BRANCH(MEMBER) \
    sim_tree_->SetBranchAddress(#MEMBER, &sim_edep_data_.MEMBER);

    LDR_SET_SIM_BRANCH(NumPhotons);
    LDR_SET_SIM_BRANCH(NumElectrons);
    LDR_SET_SIM_BRANCH(ScintYieldRatio);
    LDR_SET_SIM_BRANCH(Energy);
    LDR_SET_SIM_BRANCH(Time);
    LDR_SET_SIM_BRANCH(StartX);
    LDR_SET_SIM_BRANCH(StartY);
    LDR_SET_SIM_BRANCH(StartZ);
    LDR_SET_SIM_BRANCH(EndX);
    LDR_SET_SIM_BRANCH(EndY);
    LDR_SET_SIM_BRANCH(EndZ);
    LDR_SET_SIM_BRANCH(StartT);
    LDR_SET_SIM_BRANCH(EndT);
    LDR_SET_SIM_BRANCH(TrackID);
    LDR_SET_SIM_BRANCH(PdgCode);

#undef LDR_SET_SIM_BRANCH
}

//---------------------------------------------------------------------------//
/*!
 * Return number of events in the ROOT file.
 */
size_type LarDataReader::num_events() const
{
    return sim_tree_->GetEntries();
}

//---------------------------------------------------------------------------//
/*!
 * Read the current event's SimEnergyDeposit data from ROOT and return a
 * vector of \c sim::SimEnergyDeposit objects.
 */
LarDataReader::VecSimEdep LarDataReader::read_event(size_type event_id) const
{
    CELER_ENSURE(event_id < sim_tree_->GetEntries());

    sim_tree_->GetEntry(event_id);
    size_type const num_hits = sim_edep_data_.NumPhotons->size();

#define LDR_GET(MEMBER) (*sim_edep_data_.MEMBER)[i]

    VecSimEdep result(num_hits);
    for (auto i : range(num_hits))
    {
        sim::SimEnergyDeposit sed(
            LDR_GET(NumPhotons),
            LDR_GET(NumElectrons),
            LDR_GET(ScintYieldRatio),
            LDR_GET(Energy),
            geo::Point_t{LDR_GET(StartX), LDR_GET(StartY), LDR_GET(StartZ)},
            geo::Point_t{LDR_GET(EndX), LDR_GET(EndY), LDR_GET(EndZ)},
            LDR_GET(StartT),
            LDR_GET(EndT),
            LDR_GET(TrackID),
            LDR_GET(PdgCode));

        result[i] = std::move(sed);
    }
    return result;

#undef LDR_GET
}

//---------------------------------------------------------------------------//
/*!
 * Return detector name from ROOT file.
 */
std::string LarDataReader::detector_name() const
{
    auto* tree = data_dir_->Get<TTree>(this->detector_info_tree_name());
    CELER_ASSERT(tree);

    std::string name;
    tree->SetBranchAddress("name", &name);
    tree->GetEntry(0);
    return name;
}

//---------------------------------------------------------------------------//
/*!
 * Return vector of optical detector centers from ROOT file. The vector index
 * corresponds to the optical detector ID.
 */
LarDataReader::VecOpDetCenter LarDataReader::optical_detector_centers() const
{
    auto* tree = data_dir_->Get<TTree>(this->optical_detectors_tree_name());
    CELER_ASSERT(tree);

    Real3 pos;
    tree->SetBranchAddress("pos", &pos);
    auto const num_dets = tree->GetEntries();

    VecOpDetCenter result(num_dets);
    for (auto i : range(num_dets))
    {
        tree->GetEntry(i);
        result[i] = {pos[0], pos[1], pos[2]};
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
