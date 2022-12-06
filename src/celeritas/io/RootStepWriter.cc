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
 * Copy StepData collection values to TStepData.
 */
template<class T1, class T2>
void copy_if_selected(const T1& src, T2& dst)
{
    dst = src;
}

//---------------------------------------------------------------------------//
/*!
 * Copy pre- and post-step position and direction arrays.
 */
void copy_if_selected(const celeritas::Real3& input,
                      std::array<double, 3>&  output)
{
    std::memcpy(&output, &input, sizeof(input));
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
                               StepSelection     selection,
                               Filters           filters)
    : StepInterface()
    , root_manager_(root_manager)
    , particles_(particle_params)
    , selection_(selection)
    , filters_(filters)
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

#define RSW_STORE_PARTICLE(ATTR)                                           \
    do                                                                     \
    {                                                                      \
        if (selection_.ATTR)                                               \
        {                                                                  \
            copy_if_selected(particles_->id_to_pdg(steps.ATTR[tid]).get(), \
                             tstep_.ATTR);                                 \
        }                                                                  \
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
        RSW_STORE_PARTICLE(particle);

        for (const auto i : range(StepPoint::size_))
        {
            RSW_STORE(points[i].volume_id, .get());
            RSW_STORE(points[i].energy, .value());
            RSW_STORE(points[i].time, /* no getter */);
            RSW_STORE(points[i].dir, /* no getter */);
            RSW_STORE(points[i].pos, /* no getter */);
        }

        tstep_tree_->Fill();
    }

#undef RSW_STORE
#undef RSW_STORE_PARTICLE
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
 * To simplify the process of moving from Collection to a ROOT branch, a C++
 * macro is used.
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

    tstep_tree_.reset(new TTree("steps", "steps"));
    tstep_tree_->Branch("track_id", &tstep_.track_id); // Always on

    // Step data
    RSW_CREATE_BRANCH(event_id, "event_id");
    RSW_CREATE_BRANCH(track_step_count, "track_step_count");
    RSW_CREATE_BRANCH(action_id, "action_id");
    RSW_CREATE_BRANCH(step_length, "step_length");
    RSW_CREATE_BRANCH(particle, "particle");
    // Pre-step
    RSW_CREATE_BRANCH(points[StepPoint::pre].volume_id, "point_pre_volume_id");
    RSW_CREATE_BRANCH(points[StepPoint::pre].dir, "point_pre_dir");
    RSW_CREATE_BRANCH(points[StepPoint::pre].pos, "point_pre_pos");
    RSW_CREATE_BRANCH(points[StepPoint::pre].energy, "point_pre_energy");
    RSW_CREATE_BRANCH(points[StepPoint::pre].time, "point_pre_time");
    // Post-step
    RSW_CREATE_BRANCH(points[StepPoint::post].volume_id,
                      "point_post_volume_id");
    RSW_CREATE_BRANCH(points[StepPoint::post].dir, "point_post_dir");
    RSW_CREATE_BRANCH(points[StepPoint::post].pos, "point_post_pos");
    RSW_CREATE_BRANCH(points[StepPoint::post].energy, "point_post_energy");
    RSW_CREATE_BRANCH(points[StepPoint::post].time, "point_post_time");

#undef RSW_CREATE_BRANCH
}

//---------------------------------------------------------------------------//
} // namespace celeritas
