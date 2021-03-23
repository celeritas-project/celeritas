//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldUtils.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
// Perform y <- ax + y for OdeState
inline CELER_FUNCTION void axpy(real_type a, const OdeState& x, OdeState* y);

// Evaluate the stepper truncation error
inline CELER_FUNCTION real_type truncation_error(real_type       step,
                                                 real_type       eps_rel_max,
                                                 const OdeState& beg_state,
                                                 const OdeState& err_state);

// Closest distance between the chord and the mid-state
inline CELER_FUNCTION real_type distance_chord(const OdeState& beg_state,
                                               const OdeState& mid_state,
                                               const OdeState& end_state);

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldUtils.i.hh"
