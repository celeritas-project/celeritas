//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../Types.hh"
#include "base/Array.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Return y <- ax for a real variable
template<class T>
inline CELER_FUNCTION Array<T, 3> ax(T a, const Array<T, 3>& x);

//---------------------------------------------------------------------------//
// Calculate the direction between the source and destination
inline CELER_FUNCTION Real3 make_direction(const Real3& src, const Real3& dst);

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
} // namespace detail
} // namespace celeritas

#include "FieldUtils.i.hh"
