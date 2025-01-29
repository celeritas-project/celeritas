//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/xs/NucleonNucleonXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericCalculator.hh"
#include "celeritas/neutron/data/NeutronInelasticData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate nucleon-nucleon (NN) cross sections from NeutronInelasticData
 */
class NucleonNucleonXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NeutronInelasticRef;
    using Energy = units::MevEnergy;
    using BarnXs = units::BarnXs;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION NucleonNucleonXsCalculator(ParamsRef const& shared);

    // Compute cross section
    inline CELER_FUNCTION BarnXs operator()(ChannelId el_id,
                                            Energy energy) const;

    static CELER_CONSTEXPR_FUNCTION Energy high_otf_energy()
    {
        return Energy{10};
    }

    static CELER_CONSTEXPR_FUNCTION Energy low_otf_energy()
    {
        return Energy{1};
    }

  private:
    // Shared constant physics properties
    NeutronInelasticRef const& shared_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
NucleonNucleonXsCalculator::NucleonNucleonXsCalculator(ParamsRef const& shared)
    : shared_(shared)
{
}

//---------------------------------------------------------------------------//
/*!
 * Compute nucleon-nucleon (NN) cross section
 *
 * The parameterization of nucleon-nucleon cross sections below 10 MeV takes
 * the following functional form,
 * \f[
 *  SF(E) = coeffs[0] + coeffs[1]/E + coeffs[2]/E^{2}
 * \f]
 * where the kinetic energy of the incident nucleon, \em E is in [1, 10] MeV.
 * Below 1 MeV, \f$ SF(E) = slope/E \f$ down to \f$ E = slope/xs_zero \f$ while
 * \f$ SF(E) = xs_zero \f$ if \em E is in [0, slope/xs_zero] MeV.
 */
CELER_FUNCTION
auto NucleonNucleonXsCalculator::operator()(ChannelId ch_id,
                                            Energy energy) const -> BarnXs
{
    CELER_EXPECT(ch_id < shared_.nucleon_xs.size());
    real_type result;

    if (energy < this->high_otf_energy())
    {
        // Calculate NN cross section according to the Stepanov's function
        // for the incident nucleon kinetic energy below 10 MeV
        StepanovParameters const& par = shared_.xs_params[ch_id];

        if (energy <= this->low_otf_energy())
        {
            result = celeritas::min(par.slope / energy.value(), par.xs_zero);
        }
        else
        {
            using StepanovFunction = PolyEvaluator<real_type, 2>;
            result
                = StepanovFunction(par.coeffs)(real_type{1} / energy.value());
        }
    }
    else
    {
        // Get tabulated NN cross section data for the given channel
        GenericGridRecord grid = shared_.nucleon_xs[ch_id];

        // Calculate NN cross section at the given energy
        GenericCalculator calc_xs(grid, shared_.reals);
        result = calc_xs(energy.value());
    }

    return BarnXs{result};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
