//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootStepWriter.cc
//---------------------------------------------------------------------------//
#include "RootStepWriter.hh"

#include <algorithm>
#include <tuple>
#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Copy StepStateData collection values to TStepData.
 */
template<class T1, class T2>
void copy_if_selected(const T1& src, T2& dst)
{
    dst = src;
}

//---------------------------------------------------------------------------//
/*!
 * Copy StepPointStateData Real3 position and direction to TStepPoint arrays.
 */
void copy_if_selected(const celeritas::Real3& src, std::array<double, 3>& dst)
{
    std::memcpy(&dst, &src, sizeof(src));
}

//---------------------------------------------------------------------------//
} // namespace

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct writer with RootFileManager, ParticleParams (to convert particle
 * id to pdg), and the selection of data to be tallied.
 */
RootStepWriter::RootStepWriter(SPRootFileManager root_manager,
                               SPParticleParams  particle_params,
                               StepSelection     selection)
    : StepInterface()
    , root_manager_(root_manager)
    , particles_(particle_params)
    , selection_(selection)
{
    CELER_EXPECT(root_manager_);
    this->make_tree();
}

//---------------------------------------------------------------------------//
/*!
 * Set the number of entries (i.e. number of steps) stored in memory before
 * ROOT flushes the data to disk. Default is ~32MB of compressed data.
 *
 * See `SetAutoFlush` in ROOT TTree Class reference for details:
 * https://root.cern.ch/doc/master/classTTree.html
 */
void RootStepWriter::set_auto_flush(long num_entries)
{
    CELER_EXPECT(root_manager_);
    CELER_EXPECT(tstep_tree_);
    tstep_tree_->SetAutoFlush(num_entries);
}

//---------------------------------------------------------------------------//
/*!
 * Collect step data from each track on each thread id and fill the ROOT step
 * tree.
 */
void RootStepWriter::execute(StateHostRef const& steps)
{
#define RSW_STORE(ATTR, GETTER)                                    \
    do                                                             \
    {                                                              \
        if (selection_.ATTR)                                       \
        {                                                          \
            copy_if_selected(steps.ATTR[tid] GETTER, tstep_.ATTR); \
        }                                                          \
    } while (0)

    CELER_EXPECT(steps);
    tstep_ = TStepData();

    // Loop over thread ids and fill TTree
    for (const auto tid : range(ThreadId{steps.size()}))
    {
        if (!steps.track_id[tid])
        {
            // Track id not found; skip inactive track slot
            continue;
        }

        // Track id is always set
        tstep_.track_id = steps.track_id[tid].unchecked_get();

        RSW_STORE(event_id, .get());
        RSW_STORE(action_id, .get());
        RSW_STORE(energy_deposition, .value());
        RSW_STORE(step_length, /* no getter */);
        RSW_STORE(track_step_count, /* no getter */);
        if (selection_.particle)
        {
            copy_if_selected(particles_->id_to_pdg(steps.particle[tid]).get(),
                             tstep_.particle);
        }

        for (const auto sp : range(StepPoint::size_))
        {
            RSW_STORE(points[sp].volume_id, .get());
            RSW_STORE(points[sp].energy, .value());
            RSW_STORE(points[sp].time, /* no getter */);
            RSW_STORE(points[sp].dir, /* no getter */);
            RSW_STORE(points[sp].pos, /* no getter */);
        }

        tstep_tree_->Fill();
    }

#undef RSW_STORE
}

//---------------------------------------------------------------------------//
/*!
 * Create steps tree. In order to have the option to individually select any
 * member of `StepStateData` (StepData.hh) to be stored into the ROOT file
 * *and* not need any dictionary for ROOT I/O, we cannot store an MC truth step
 * object. Therefore, the data is flattened so that each member of `TStepData`
 * is an individual branch that stores primitive types and is created based on
 * the `StepSelection` booleans.
 *
 * A macro is usedT to simplify the process of moving from Collection to a ROOT
 * branch.
 */
void RootStepWriter::make_tree()
{
#define RSW_CREATE_BRANCH(ATTR, BRANCH_NAME)                      \
    do                                                            \
    {                                                             \
        if (this->selection_.ATTR)                                \
        {                                                         \
            this->tstep_tree_->Branch(BRANCH_NAME, &tstep_.ATTR); \
        }                                                         \
    } while (0)
#define RSW_CONCAT(ONE, TWO) ONE TWO

    tstep_tree_.reset(new TTree("steps", "steps"));

    tstep_tree_->Branch("track_id", &tstep_.track_id); // Always on
    RSW_CREATE_BRANCH(event_id, "event_id");
    RSW_CREATE_BRANCH(track_step_count, "track_step_count");
    RSW_CREATE_BRANCH(action_id, "action_id");
    RSW_CREATE_BRANCH(step_length, "step_length");
    RSW_CREATE_BRANCH(particle, "particle");

    const EnumArray<StepPoint, std::string> pref{{"pre_", "post_"}};
    for (const auto sp : range(StepPoint::size_))
    {
        RSW_CREATE_BRANCH(points[sp].volume_id, pref[sp] + "volume_id");
        RSW_CREATE_BRANCH(points[sp].dir, pref[sp] + "dir");
        RSW_CREATE_BRANCH(points[sp].pos, pref[sp] + "pos");
        RSW_CREATE_BRANCH(points[sp].energy, pref[sp] + "energy");
        RSW_CREATE_BRANCH(points[sp].time, pref[sp] + "time");
    }

#undef RSW_CREATE_BRANCH
}

//---------------------------------------------------------------------------//
} // namespace celeritas
