//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollectorTestBase.cc
//---------------------------------------------------------------------------//
#include "StepCollectorTestBase.hh"

#include "corecel/io/LogContextException.hh"
#include "celeritas/global/Stepper.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Run a stepping loop with the core data.
 */
template<MemSpace M>
void StepCollectorTestBase::run_impl(size_type num_tracks_per_batch,
                                     size_type num_steps,
                                     size_type num_batches)
{
    // Save number of batches internally for normalization
    num_batches_ = num_batches;

    // Compute total tracks
    size_type num_tracks = num_tracks_per_batch * num_batches;

    // Initialize stepper
    StepperInput step_inp;
    step_inp.params = this->core();
    step_inp.stream_id = StreamId{0};
    step_inp.num_track_slots = num_tracks;
    Stepper<M> step(step_inp);
    LogContextException log_context{this->output_reg().get()};

    // Initalize results
    this->initalize();

    // Loop over batches
    for (size_type i_batch = 0; i_batch < num_batches; ++i_batch)
    {
        // Initialize primaries for this batch
        auto primaries = this->make_primaries(num_tracks_per_batch);

        // Take num_steps steps
        StepperResult count;
        CELER_TRY_HANDLE(count = step(make_span(primaries)), log_context);
        while (count && --num_steps > 0)
        {
            CELER_TRY_HANDLE(count = step(), log_context);
        }

        // Gathering of results
        this->gather_batch_results();
    }

    // Post-processing (e.g. normalization) of results
    this->finalize();
}

template void StepCollectorTestBase::run_impl<MemSpace::host>(size_type,
                                                              size_type,
                                                              size_type);
template void StepCollectorTestBase::run_impl<MemSpace::device>(size_type,
                                                                size_type,
                                                                size_type);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
