//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/RecursiveSimplifierImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <variant>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Recursively simplify, then call the given function on the final surface.
 *
 * This implementation class allows std::visit to be used with the surface
 * while retaining updates to the associated Sense that the user requested.
 */
template<class F>
class RecursiveSimplifierImpl
{
  public:
    //! Construct with reference to function and values to be used
    RecursiveSimplifierImpl(F& func, Sense sense, real_type tol)
        : func_{func}, sense_{sense}, simplify_{&sense_, tol}
    {
    }

    //! Invoke recursively with a specific surface type
    template<class S>
    void operator()(S const& surf)
    {
        auto result = simplify_(surf);
        CELER_ASSUME(!result.valueless_by_exception());
        if (std::holds_alternative<std::monostate>(result))
        {
            // Could not simplify further: call back with sense and surface
            return func_(sense_, surf);
        }
        else
        {
            return std::visit(*this, result);
        }
    }

    //! Monostate should never be reached
    void operator()(std::monostate) { CELER_ASSERT_UNREACHABLE(); }

  private:
    F& func_;
    Sense sense_;
    SurfaceSimplifier simplify_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
