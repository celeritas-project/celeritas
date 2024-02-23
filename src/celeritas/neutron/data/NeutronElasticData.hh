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
 * Model and particles IDs for neutron--nucleus elastic scattering.
 */
struct NeutronElasticIds
{
    ActionId action;
    ParticleId neutron;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const { return action && neutron; }
};

//---------------------------------------------------------------------------//
/*!
 * Microscopic cross section data for the neutron elastic interaction.
 */
template<Ownership W, MemSpace M>
struct NeutronElasticXsData
{
    using XsUnits = units::Native;  // [len^2]

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    Items<real_type> reals;
    ElementItems<GenericGridData> elements;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !elements.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    NeutronElasticXsData& operator=(NeutronElasticXsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        reals = other.reals;
        elements = other.elements;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * A-dependent data for the differential cross section (momentum transfer) of
 * the CHIPS neutron-nucleus elastic model.
 */
template<Ownership W, MemSpace M>
struct NeutronElasticDiffXsData
{
    template<class T>
    using IsotopeItems = Collection<T, W, M, IsotopeId>;

    //// MEMBER DATA ////

    IsotopeItems<ChipsDiffXsCoefficients> coeffs;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !coeffs.empty() && !coeffs.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    NeutronElasticDiffXsData&
    operator=(NeutronElasticDiffXsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        coeffs = other.coeffs;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
template<Ownership W, MemSpace M>
struct NeutronElasticData
{
    //! Model and particle IDs
    NeutronElasticIds ids;

    //! Particle mass * c^2 [MeV]
    units::MevMass neutron_mass;

    //! Neutron elastic cross section (G4PARTICLEXS/neutron/elZ) data
    NeutronElasticXsData<W, M> xs;

    //! Neutron elastic differential cross section coefficients
    NeutronElasticDiffXsData<W, M> diff_xs;

    //// MEMBER FUNCTIONS ////

    //! Model's minimum and maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy mim_valid_energy()
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
        return ids && neutron_mass > zero_quantity();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    NeutronElasticData& operator=(NeutronElasticData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        neutron_mass = other.neutron_mass;
        xs = other.xs;
        diff_xs = other.diff_xs;
        return *this;
    }
};

using NeutronElasticHostRef = HostCRef<NeutronElasticData>;
using NeutronElasticDeviceRef = DeviceCRef<NeutronElasticData>;
using NeutronElasticRef = NativeCRef<NeutronElasticData>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
