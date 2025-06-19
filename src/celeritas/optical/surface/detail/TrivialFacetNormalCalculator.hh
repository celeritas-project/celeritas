//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/TrivialFacetNormalCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    TrivialFacetNormalCalculator ...;
   \endcode
 */
struct TrivialFacetNormalCalculator
{
    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
