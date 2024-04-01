//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/data/NeutronElasticData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Parameters for sampling the momentum transfer of CHIPS neutron-nucleus
 * elastic scattering.
 */
struct ExchangeParameters
{
    using Real4 = Array<real_type, 4>;

    // Momentum-dependent parameters in G4ChipsNeutronElasticXS
    real_type ss{};  // square slope of the first diffractive maximum
    Real4 slope{0, 0, 0, 0};  //!< slope of CHIPS diffractive maxima
    Real4 expnt{0, 0, 0, 0};  //!< mantissa of CHIPS diffractive maxima
};

//---------------------------------------------------------------------------//
/*!
 * A-dependent data for the differential cross section (momentum transfer) of
 * the CHIPS neutron-nucleus elastic model.
 */
struct ChipsDiffXsCoefficients
{
    using ChipsArray = Array<real_type, 42>;

    // Coefficients
    ChipsArray par{};  //!< Coefficients as a function of atomic mass numbers
};

//---------------------------------------------------------------------------//
/*!
 * Scalar data for neutron-nucleus elastic scattering.
 */
struct NeutronElasticScalars
{
    //! Action and particle IDs
    ActionId action_id;
    ParticleId neutron_id;

    //! Particle mass * c^2 [MeV]
    units::MevMass neutron_mass;

    //! Model's minimum and maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_valid_energy()
    {
        return units::MevEnergy{1e-5};
    }

    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        return units::MevEnergy{2e+4};
    }

    //! Whether data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return action_id && neutron_id && neutron_mass > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
template<Ownership W, MemSpace M>
struct NeutronElasticData
{
    using XsUnits = units::Native;  // [len^2]

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;
    template<class T>
    using IsotopeItems = Collection<T, W, M, IsotopeId>;

    //// MEMBER DATA ////

    //! Scalar data
    NeutronElasticScalars scalars;

    //! Microscopic (element) cross section data (G4PARTICLEXS/neutron/elZ)
    ElementItems<GenericGridData> micro_xs;

    //! A-dependent coefficients for the momentum transfer of the CHIPS model
    IsotopeItems<ChipsDiffXsCoefficients> coeffs;

<<<<<<< HEAD
    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Model's minimum and maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_valid_energy()
    {
        return units::MevEnergy{1e-5};
    }

    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        return units::MevEnergy{2e+4};
    }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && neutron_mass > zero_quantity() && !micro_xs.empty()
               && !coeffs.empty() && !reals.empty();
=======
    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !reals.empty() && !micro_xs.empty()
               && !coeffs.empty();
>>>>>>> 8a25bcfe2 (Change NeutronElasticIds to NeutronElasticScalars and move other scalar data to NeutronElasticScalars)
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    NeutronElasticData& operator=(NeutronElasticData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
<<<<<<< HEAD
        ids = other.ids;
        neutron_mass = other.neutron_mass;
=======
        scalars = other.scalars;
        reals = other.reals;
>>>>>>> 8a25bcfe2 (Change NeutronElasticIds to NeutronElasticScalars and move other scalar data to NeutronElasticScalars)
        micro_xs = other.micro_xs;
        coeffs = other.coeffs;
        reals = other.reals;
        return *this;
    }
};

using NeutronElasticHostRef = HostCRef<NeutronElasticData>;
using NeutronElasticDeviceRef = DeviceCRef<NeutronElasticData>;
using NeutronElasticRef = NativeCRef<NeutronElasticData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
