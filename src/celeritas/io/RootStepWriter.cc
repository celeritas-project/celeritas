//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootStepWriter.cc
//---------------------------------------------------------------------------//
#include "RootStepWriter.hh"

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
void copy_if_selected(celeritas::Real3 const& src, std::array<double, 3>& dst)
{
    std::memcpy(&dst, &src, sizeof(src));
}

//---------------------------------------------------------------------------//
}  // namespace

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct writer with RootFileManager, ParticleParams (to convert particle
 * id to pdg), and the selection of data to be tallied.
 */
RootStepWriter::RootStepWriter(SPRootFileManager root_manager,
                               SPParticleParams particle_params,
                               StepSelection selection,
                               RSWFilter filter_conditions)
    : StepInterface()
    , root_manager_(root_manager)
    , particles_(particle_params)
    , selection_(selection)
{
    CELER_EXPECT(root_manager_);
    rsw_filter_ = std::make_unique<RSWFilter>(filter_conditions);
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
 * Collect step data and fill the ROOT TTree for all active threads.
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

#define RSW_FILTER(ATTR, GETTER)                                        \
    do                                                                  \
    {                                                                   \
        if (!rsw_filter_)                                               \
        {                                                               \
            RSW_STORE(ATTR, GETTER);                                    \
        }                                                               \
                                                                        \
        else if (rsw_filter_->first.ATTR                                \
                 && rsw_filter_->second.ATTR == steps.ATTR[tid] GETTER) \
        {                                                               \
            RSW_STORE(ATTR, GETTER);                                    \
        }                                                               \
    } while (0)

    CELER_EXPECT(steps);
    tstep_ = TStepData();

    // Loop over thread ids and fill TTree
    for (auto const tid : range(ThreadId{steps.size()}))
    {
        if (!steps.track_id[tid])
        {
            // Track id not found; skip inactive track slot
            continue;
        }

        // Track id is always set
        tstep_.track_id = steps.track_id[tid].unchecked_get();

        RSW_FILTER(event_id, .get());
        RSW_FILTER(parent_id, .unchecked_get());
        RSW_FILTER(action_id, .get());
        RSW_FILTER(energy_deposition, .value());
        RSW_FILTER(step_length, /* no getter */);
        RSW_FILTER(track_step_count, /* no getter */);
        if (selection_.particle)
        {
            auto const pdg = particles_->id_to_pdg(steps.particle[tid]).get();

            if (rsw_filter_->first.particle
                && rsw_filter_->second.particle == pdg)
            {
                copy_if_selected(pdg, tstep_.particle);
            }
        }

        for (auto const sp : range(StepPoint::size_))
        {
            RSW_STORE(points[sp].volume_id, .unchecked_get());
            RSW_STORE(points[sp].energy, .value());
            RSW_STORE(points[sp].time, /* no getter */);
            RSW_STORE(points[sp].dir, /* no getter */);
            RSW_STORE(points[sp].pos, /* no getter */);
        }

        tstep_tree_->Fill();
    }

#undef RSW_STORE
#undef RSW_FILTER
}

//---------------------------------------------------------------------------//
/*!
 * Create steps tree. In order to have the option to individually select any
 * member of `StepStateData` (StepData.hh) to be stored into the ROOT file
 * *and* not need any dictionary for ROOT I/O, we cannot store an MC truth step
 * object. Therefore, the data is flattened so that each member of `TStepData`
 * is an individual branch that stores primitive types and is created based on
 * the `StepSelection` booleans.
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

    tstep_tree_ = root_manager_->make_tree("steps", "steps");

    tstep_tree_->Branch("track_id", &tstep_.track_id);  // Always on
    RSW_CREATE_BRANCH(event_id, "event_id");
    RSW_CREATE_BRANCH(parent_id, "parent_id");
    RSW_CREATE_BRANCH(track_step_count, "track_step_count");
    RSW_CREATE_BRANCH(action_id, "action_id");
    RSW_CREATE_BRANCH(step_length, "step_length");
    RSW_CREATE_BRANCH(particle, "particle");
    RSW_CREATE_BRANCH(energy_deposition, "energy_deposition");
    // Pre-step
    RSW_CREATE_BRANCH(points[StepPoint::pre].volume_id, "pre_volume_id");
    RSW_CREATE_BRANCH(points[StepPoint::pre].dir, "pre_dir");
    RSW_CREATE_BRANCH(points[StepPoint::pre].pos, "pre_pos");
    RSW_CREATE_BRANCH(points[StepPoint::pre].energy, "pre_energy");
    RSW_CREATE_BRANCH(points[StepPoint::pre].time, "pre_time");
    // Post-step
    RSW_CREATE_BRANCH(points[StepPoint::post].volume_id, "post_volume_id");
    RSW_CREATE_BRANCH(points[StepPoint::post].dir, "post_dir");
    RSW_CREATE_BRANCH(points[StepPoint::post].pos, "post_pos");
    RSW_CREATE_BRANCH(points[StepPoint::post].energy, "post_energy");
    RSW_CREATE_BRANCH(points[StepPoint::post].time, "post_time");

#undef RSW_CREATE_BRANCH
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
