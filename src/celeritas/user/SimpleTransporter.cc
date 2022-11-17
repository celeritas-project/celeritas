//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SimpleTransporter.cc
//---------------------------------------------------------------------------//
#include "SimpleTransporter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
SimpleTransporterInterface::~SimpleTransporterInterface() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct the stepper for reuse on consecutive calls.
 */
template<MemSpace M>
SimpleTransporter<M>::SimpleTransporter(StepperInput           input,
                                        SimpleTransporterInput st_inp)
    : step_(std::move(input)), st_input_(st_inp)
{
    CELER_ENSURE(step_);
}

//---------------------------------------------------------------------------//
/*!
 * Transport the input primaries and all secondaries produced
 */
template<MemSpace M>
void SimpleTransporter<M>::operator()(const VecPrimary& primaries)
{
    // Copy primaries to device and transport the first step
    auto track_counts = step_(primaries);

    size_type step_iters = 1;

    while (track_counts)
    {
        CELER_VALIDATE(step_iters < st_input_.max_step_iterations,
                       << "number of step iterations exceeded the allowed "
                          "maximum ("
                       << st_input_.max_step_iterations << ")");

        track_counts = step_();
        step_iters += 1;
    }
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class SimpleTransporter<MemSpace::host>;
template class SimpleTransporter<MemSpace::device>;

//---------------------------------------------------------------------------//
} // namespace celeritas
