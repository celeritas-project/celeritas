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

//---------------------------------------------------------------------------//
// Free helper functions
//---------------------------------------------------------------------------//

namespace
{
//---------------------------------------------------------------------------//
/*!
 * Copy pre- and post-step position and direction arrays.
 */
void copy_real3(const celeritas::Real3& input, std::array<double, 3>& output)
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
 * Collect step data from each track on each thread id.
 */
void RootStepWriter::execute(StateHostRef const& steps)
{
#define IF_SELECTED(ATTR, REF, VAL) \
    do                              \
    {                               \
        if (selection_.ATTR)        \
        {                           \
            tstep_.REF = VAL;       \
        }                           \
    } while (0)

#define IF_SELECTED_REAL3(ATTR, REF)                 \
    do                                               \
    {                                                \
        if (selection_.ATTR)                         \
        {                                            \
            copy_real3(steps.ATTR[tid], tstep_.REF); \
        }                                            \
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

        IF_SELECTED(event_id, event_id, steps.event_id[tid].get());
        IF_SELECTED(action_id, action_id, steps.action_id[tid].get());
        IF_SELECTED(particle,
                    particle,
                    particles_->id_to_pdg(steps.particle[tid]).get());
        IF_SELECTED(energy_deposition,
                    energy_deposition,
                    steps.energy_deposition[tid].value());
        IF_SELECTED(step_length, step_length, steps.step_length[tid]);
        IF_SELECTED(
            track_step_count, track_step_count, steps.track_step_count[tid]);

        // Store pre-step
        const auto& pre = steps.points[StepPoint::pre];
        IF_SELECTED(points[StepPoint::pre].volume_id,
                    point_pre_volume_id,
                    pre.volume_id[tid].get());
        IF_SELECTED(points[StepPoint::pre].energy,
                    point_pre_energy,
                    pre.energy[tid].value());
        IF_SELECTED(points[StepPoint::pre].time, point_pre_time, pre.time[tid]);
        IF_SELECTED_REAL3(points[StepPoint::pre].dir, point_pre_dir);
        IF_SELECTED_REAL3(points[StepPoint::pre].pos, point_pre_pos);

        // Store post-step
        const auto& post = steps.points[StepPoint::post];
        IF_SELECTED(points[StepPoint::post].volume_id,
                    point_post_volume_id,
                    post.volume_id[tid].get());
        IF_SELECTED(points[StepPoint::post].energy,
                    point_post_energy,
                    post.energy[tid].value());
        IF_SELECTED(
            points[StepPoint::post].time, point_post_time, post.time[tid]);
        IF_SELECTED_REAL3(points[StepPoint::post].dir, point_post_dir);
        IF_SELECTED_REAL3(points[StepPoint::post].pos, point_post_pos);

        tstep_tree_->Fill();
    }

#undef IF_SELECTED
#undef IF_SELECTED_REAL3
}

//---------------------------------------------------------------------------//
/*!
 * Create steps tree. In order to have the option to individually select any
 * member of `StepStateData` (StepData.hh) to be stored into the ROOT file
 * *and* not need any dictionary for ROOT I/O, we cannot store an MC truth step
 * object. This is accomplished by "flattening" the data so that each member of
 * `TStepData`is an individual branch, constructed with primitive types, that
 * can be created based on the `StepSelection` booleans.
 *
 * To simplify the process of moving from Collection to a ROOT branch
 * `TStepData`, a C++ macro is used.
 */
void RootStepWriter::make_tree()
{
#define CREATE_BRANCH_IF_SELECTED(ATTR, REF)              \
    do                                                    \
    {                                                     \
        if (this->selection_.ATTR)                        \
        {                                                 \
            this->tstep_tree_->Branch(#REF, &tstep_.REF); \
        }                                                 \
    } while (0)

    tstep_tree_.reset(new TTree("steps", "steps"));
    tstep_tree_->Branch("track_id", &tstep_.track_id); // Always on

    // Step selection data
    CREATE_BRANCH_IF_SELECTED(event_id, event_id);
    CREATE_BRANCH_IF_SELECTED(track_step_count, track_step_count);
    CREATE_BRANCH_IF_SELECTED(action_id, action_id);
    CREATE_BRANCH_IF_SELECTED(step_length, step_length);
    CREATE_BRANCH_IF_SELECTED(particle, particle);
    // Pre-step
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::pre].volume_id,
                              point_pre_volume_id);
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::pre].dir, point_pre_dir);
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::pre].pos, point_pre_pos);
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::pre].energy, point_pre_energy);
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::pre].time, point_pre_time);
    // Post-step
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::post].volume_id,
                              point_post_volume_id);
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::post].dir, point_post_dir);
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::post].pos, point_post_pos);
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::post].energy,
                              point_post_energy);
    CREATE_BRANCH_IF_SELECTED(points[StepPoint::post].time, point_post_time);

#undef CREATE_BRANCH_IF_SELECTED
}

//---------------------------------------------------------------------------//
} // namespace celeritas
