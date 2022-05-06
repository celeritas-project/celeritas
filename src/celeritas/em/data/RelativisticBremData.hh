//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/RelativisticBremData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/Types.hh"
#include "celeritas/Quantities.hh"

#include "ElectronBremsData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * The atomic form factors used in the differential cross section of the
 * bremsstrahlung process by an ultrarelativistic electron.
 */
struct RelBremFormFactor
{
    real_type el;   //!< elastic component
    real_type inel; //!< inelastic component
};

//---------------------------------------------------------------------------//
/*!
 * A special metadata structure per element used in the differential cross
 * section calculation.
 */
struct RelBremElementData
{
    real_type fz;             //!< \f$ \ln(Z)/3 + f_c (Coulomb correction) \f$
    real_type factor1;        //!< \f$ ((Fel-fc)+Finel*invZ)\f$
    real_type factor2;        //!< \f$ (1.0+invZ)/12 \f$
    real_type gamma_factor;   //!< Constant for evaluating screening functions
    real_type epsilon_factor; //!< Constant for evaluating screening functions
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
template<Ownership W, MemSpace M>
struct RelativisticBremData
{
    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    //! IDs
    ElectronBremIds ids;

    //! Electron mass [MeVMass]
    units::MevMass electron_mass;

    //! LPM flag
    bool enable_lpm{};

    //! Element data
    ElementItems<RelBremElementData> elem_data;

    //// MEMBER FUNCTIONS ////

    //! Include a dielectric suppression effect in LPM functions
    static CELER_CONSTEXPR_FUNCTION bool dielectric_suppression()
    {
        return true;
    }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity() && !elem_data.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RelativisticBremData& operator=(const RelativisticBremData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        ids           = other.ids;
        electron_mass = other.electron_mass;
        enable_lpm    = other.enable_lpm;
        elem_data     = other.elem_data;
        return *this;
    }
};

using RelativisticBremDeviceRef
    = RelativisticBremData<Ownership::const_reference, MemSpace::device>;
using RelativisticBremHostRef
    = RelativisticBremData<Ownership::const_reference, MemSpace::host>;
using RelativisticBremRef
    = RelativisticBremData<Ownership::const_reference, MemSpace::native>;

} // namespace detail
} // namespace celeritas
