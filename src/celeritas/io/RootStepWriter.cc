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
#include <TTree.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct writer with RootFileManager, ParticleParams (to convert particle
 * id to pdg), and the selection of data to be tallied.
 */
RootStepWriter::RootStepWriter(SPRootFileManager io_manager,
                               SPParticleParams  particle_params,
                               StepSelection     selection)
    : StepInterface()
    , root_manager_(io_manager)
    , particles_(particle_params)
    , selection_(selection)
{
    CELER_EXPECT(root_manager_);
    tstep_tree_.reset(new TTree("steps", "steps"));
    tstep_tree_->SetBranchAddress("steps,", &tstep_);
}

//---------------------------------------------------------------------------//
/*!
 * Set the number of entries (i.e. number of steps) stored in memory before
 * ROOT flushes the data to disk. Default is ~32MB of compressed data.
 *
 * See ROOT TTree Class reference for details:
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
 *
 * TODO std::sort data by event and track ids would be ideal, but that would
 * entail a much larger cost: copy to a host array, sort it, then loop over it
 * in order to call \c tree->Fill() .
 */
void RootStepWriter::execute(StateHostRef const& steps)
{
    CELER_EXPECT(steps);
    tstep_ = mctruth::TStepData();

    // Loop over thread ids and fill TTree
    for (const auto tid : range(ThreadId{steps.size()}))
    {
        if (!steps.track[tid])
        {
            // Track id not found; skip inactive track slot
            continue;
        }

        tstep_.event_id  = steps.event[tid].get();
        tstep_.track_id  = steps.track[tid].unchecked_get();
        tstep_.action_id = steps.action[tid].get();
        tstep_.pdg       = particles_->id_to_pdg(steps.particle[tid]).get();
        tstep_.energy_deposition = steps.energy_deposition[tid].value();
        tstep_.length            = steps.step_length[tid];
        tstep_.track_step_count  = steps.track_step_count[tid];

        // Loop for storing pre- and post-step
        for (auto i : range(StepPoint::size_))
        {
            tstep_.points[(int)i].volume_id = steps.points[i].volume[tid].get();

            // Loop over x, y, z values
            for (auto j : range(3))
            {
                tstep_.points[(int)i].dir[j] = steps.points[i].dir[tid][j];
                tstep_.points[(int)i].pos[j] = steps.points[i].pos[tid][j];
            }
        }

        tstep_tree_->Fill();
    }
}

void RootStepWriter::execute(StateDeviceRef const& steps)
{
    CELER_NOT_IMPLEMENTED("RootStepWriter is host-only");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
