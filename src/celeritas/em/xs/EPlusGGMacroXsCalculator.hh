//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/EPlusGGMacroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "celeritas/Constants.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/mat/MaterialView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculates the macroscopic cross section of positron annihilation.
 *
 * The Heitler formula (section 10.3.2 of \cite{g4prm} ) is used to compute the
 * macroscopic cross section for positron annihilation on the fly at the given
 * energy.
 */
class EPlusGGMacroXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using XsUnits = units::Native;  // [1/len]
    //!@}

  public:
    // Construct with shared data and material
    inline CELER_FUNCTION
    EPlusGGMacroXsCalculator(EPlusGGData const& shared,
                             MaterialView const& material);

    // Compute cross section on the fly at the given energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

    //! Minimum energy for Heitler formula validity
    static CELER_CONSTEXPR_FUNCTION Energy min_energy()
    {
        return units::MevEnergy{1e-6};
    }

  private:
    real_type const electron_mass_;
    real_type const electron_density_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with material.
 */
CELER_FUNCTION
EPlusGGMacroXsCalculator::EPlusGGMacroXsCalculator(EPlusGGData const& shared,
                                                   MaterialView const& material)
    : electron_mass_(value_as<units::MevMass>(shared.electron_mass))
    , electron_density_(material.electron_density())
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute macroscopic cross section in native units.
 */
CELER_FUNCTION real_type EPlusGGMacroXsCalculator::operator()(Energy energy) const
{
    using constants::pi;
    using constants::r_electron;
    using PolyQuad = PolyEvaluator<real_type, 2>;

    real_type const gamma
        = celeritas::max(energy.value(), value_as<Energy>(this->min_energy()))
          / electron_mass_;
    real_type const sqrt_gg2 = std::sqrt(gamma * (gamma + 2));

    real_type result
        = pi * ipow<2>(r_electron) * electron_density_
          * (PolyQuad{1, 4, 1}(gamma + 1) * std::log(gamma + 1 + sqrt_gg2)
             - (gamma + 4) * sqrt_gg2)
          / (gamma * ipow<2>(gamma + 2));

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
