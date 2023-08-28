//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/RecursiveSimplifier.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <variant>

#include "SurfaceSimplifier.hh"
#include "VariantSurface.hh"
#include "detail/AllSurfaces.hh"
#include "detail/RecursiveSimplifierImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Recursively simplify, then call the given function on the final surface.
 *
 * Example:
 * \code
  RecursiveSimplifier print_simplified([](Sense s, auto&& surf) {
      cout << to_char(s) << surf << endl;
  }, 1e-10);
  // Invoke on a compile-time surface type
  print_simplified(Sense::inside, Plane{{1,0,0}, 4});
  // Invoke on a run-time surface type
  for (auto&& [sense, surf] : my_senses_and_variants)
  {
      print_simplified(sense, surf);
  }
  \endcode
 *
 */
template<class F>
class RecursiveSimplifier
{
  public:
    // Construct with tolerance and function to apply
    inline RecursiveSimplifier(F&& func, real_type tol);

    // Apply to a surface with a known type
    template<class S>
    void operator()(Sense s, S const& surf);

    // Apply to a surface with unknown type
    void operator()(Sense s, VariantSurface const& surf);

  private:
    F func_;
    real_type tol_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class F, class... Ts>
RecursiveSimplifier(F&&, Ts...) -> RecursiveSimplifier<F>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with tolerance and function to apply.
 */
template<class F>
RecursiveSimplifier<F>::RecursiveSimplifier(F&& func, real_type tol)
    : func_{std::forward<F>(func)}, tol_{tol}
{
    CELER_EXPECT(tol_ >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Apply to a surface with a known type.
 */
template<class F>
template<class S>
void RecursiveSimplifier<F>::operator()(Sense sense, S const& surf)
{
    return detail::RecursiveSimplifierImpl<F>{func_, sense, tol_}(surf);
}

//---------------------------------------------------------------------------//
/*!
 * Apply to a surface with an unknown type.
 */
template<class F>
void RecursiveSimplifier<F>::operator()(Sense sense, VariantSurface const& surf)
{
    return std::visit(detail::RecursiveSimplifierImpl<F>{func_, sense, tol_},
                      surf);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
