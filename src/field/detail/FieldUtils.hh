//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "field/Types.hh"

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
struct Chord
{
    real_type length;
    Real3     dir;
};
inline CELER_FUNCTION Chord make_chord(const Real3& src, const Real3& dst);

//---------------------------------------------------------------------------//
inline CELER_FUNCTION bool is_intercept_close(const Real3& pos,
                                              const Real3& dir,
                                              real_type    distance,
                                              const Real3& target,
                                              real_type    tolerance);

//---------------------------------------------------------------------------//
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
