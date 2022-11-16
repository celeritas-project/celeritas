//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootIO.cc
//---------------------------------------------------------------------------//
#include "RootIO.hh"

#include <TFile.h>
#include <TTree.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with ROOT output filename and initialize the step TTree.
 */
RootIO::RootIO(const char* filename, SPParticleParams particles)
    : particles_(particles)
{
    CELER_EXPECT(strlen(filename));
    tfile_.reset(TFile::Open(filename, "recreate"));
    CELER_ENSURE(tfile_->IsOpen());

    step_tree_.reset(new TTree("steps", "steps"));
    step_tree_->SetBranchAddress("step", &tstep_);
}

//---------------------------------------------------------------------------//
/*!
 * Destruct by closing ROOT file.
 */
RootIO::~RootIO()
{
    if (tfile_->IsOpen())
    {
        // this->close() not invoked by user
        tfile_->Write();
        tfile_->Close();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Store step data in the TTree. This function only fills the tree, and thus
 * the user *must* invoke \c write() when appropriate to flush in-memory data
 * to disk.
 */
void RootIO::store(HostCRef<StepStateData> steps)
{
    CELER_EXPECT(steps);
    tstep_ = TStepData();

    // Loop over thread ids and fill TTree
    for (auto tid : range(ThreadId{steps.size()}))
    {
        tstep_.event_id  = steps.event[tid].get();
        tstep_.track_id  = steps.track[tid].get();
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

        step_tree_->Fill();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Set the number of entries (i.e. number of steps) stored in memory before
 * ROOT flushes the data to disk. Default is ~32MB of compressed data.
 *
 * See ROOT TTree Class reference for further details.
 */
void RootIO::set_auto_flush(long num_entries)
{
    CELER_EXPECT(tfile_->IsOpen());
    CELER_EXPECT(step_tree_);
    step_tree_->SetAutoFlush(num_entries);
}

//---------------------------------------------------------------------------//
/*!
 * View TFile pointer. Allows some extra flexibility, such as writing an input
 * ttree to it without expanding this class.
 */
TFile* RootIO::tfile_get()
{
    CELER_EXPECT(tfile_->IsOpen());
    return tfile_.get();
}

//---------------------------------------------------------------------------//
/*!
 * Write current TTree content to disk. Can be called multiple times.
 */
void RootIO::close()
{
    CELER_EXPECT(tfile_->IsOpen());
    CELER_ENSURE(tfile_->Write());
    tfile_->Close();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
