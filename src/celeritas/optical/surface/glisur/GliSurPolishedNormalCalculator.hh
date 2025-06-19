//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurPolishedNormalCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    GliSurPolishedNormalCalculator ...;
   \endcode
 */
class GliSurPolishedNormalCalculator
{
  public:
    inline CELER_FUNCTION GliSurPolishedNormalCalculator(Real3 surface_normal,
                                                         real_type polish,
                                                         Real3 inc_dir);

    template<class Engine>
    inline CELER_FUNCTION Real3 operator()(Engine& rng) const;

  private:
    Real3 normal_;
    real_type polish_;
    Real3 inc_dir_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
GliSurPolishedNormalCalculator::GliSurPolishedNormalCalculator() {}

//---------------------------------------------------------------------------//
}  // namespace celeritas
