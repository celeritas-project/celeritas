//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/LivermorePEMicroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/grid/GenericCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate photoelectric effect cross sections using the Livermore data.
 */
class LivermorePEMicroXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = LivermorePERef;
    using Energy = RealQuantity<LivermoreSubshell::EnergyUnits>;
    using BarnXs = units::BarnXs;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    LivermorePEMicroXsCalculator(ParamsRef const& shared, Energy energy);

    // Compute cross section
    inline CELER_FUNCTION BarnXs operator()(ElementId el_id) const;

  private:
    // Shared constant physics properties
    LivermorePERef const& shared_;
    // Incident gamma energy
    Energy const inc_energy_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION LivermorePEMicroXsCalculator::LivermorePEMicroXsCalculator(
    ParamsRef const& shared, Energy energy)
    : shared_(shared), inc_energy_(energy.value())
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute cross section.
 */
CELER_FUNCTION
auto LivermorePEMicroXsCalculator::operator()(ElementId el_id) const -> BarnXs
{
    CELER_EXPECT(el_id);
    LivermoreElement const& el = shared_.xs.elements[el_id];
    auto const& shells = shared_.xs.shells[el.shells];

    // In Geant4, if the incident gamma energy is below the lowest binding
    // energy, it is set to the binding energy so that the photoelectric cross
    // section is constant rather than zero for low energy gammas.
    Energy energy = max(inc_energy_, shells.back().binding_energy);
    real_type inv_energy = 1. / energy.value();

    real_type result = 0.;
    if (energy >= el.thresh_lo)
    {
        // Fit parameters from the final shell are used to calculate the cross
        // section integrated over all subshells
        auto const& param = shells.back().param[energy < el.thresh_hi ? 0 : 1];
        PolyEvaluator<real_type, 5> eval_poly(param);

        // Use the parameterization of the integrated subshell cross sections
        result = inv_energy * eval_poly(inv_energy);
    }
    else if (energy >= shells.front().binding_energy)
    {
        // Use tabulated cross sections above K-shell energy but below energy
        // limit for parameterization
        GenericCalculator calc_xs(el.xs_hi, shared_.xs.reals);
        result = ipow<3>(inv_energy) * calc_xs(energy.value());
    }
    else
    {
        CELER_ASSERT(el.xs_lo);
        // Use tabulated cross sections below K-shell energy
        GenericCalculator calc_xs(el.xs_lo, shared_.xs.reals);
        result = ipow<3>(inv_energy) * calc_xs(energy.value());
    }
    return BarnXs{result};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
