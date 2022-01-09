//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RelativisticBremData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

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
 * Evaluated LPM data for a single energy gridpoint.
 *
 * These are  \f$ G(s) \f$ and \f$ \phi(s) \f$ in a table in the range \f$ s =
 * [0, s_\mathrm{max}] \f$ with an interval \f$ \delta \f$ where \f$
 * s_\mathrm{max} = 2.0 \f$ and \f$ \delta = 0.01 \f$ by default.
 *
 * This is used by \c RBDiffXsCalculator.
 */
struct RelBremMigdalData
{
    real_type gs;   //!< LPM \f$ G(s) \f$
    real_type phis; //!< LPM \f$ \phi(s) \f$
};

//---------------------------------------------------------------------------//
/*!
 * A special metadata structure per element used in the differential cross
 * section calculation.
 */
struct RelBremElementData
{
    int       iz;             //!< Atomic number
    real_type logz;           //!< \f$ \ln(Z) \f$
    real_type fz;             //!< \f$ \ln(Z)/3 + f_c (Coulomb correction) \f$
    real_type factor1;        //!< \f$ ((Fel-fc)+Finel*invZ)\f$
    real_type factor2;        //!< \f$ (1.0+invZ)/12 \f$
    real_type s1;             //!< LPM variables
    real_type inv_logs1;      //!< \f$ 1/\ln(s1) \f$
    real_type inv_logs2;      //!< \f$ 1/\ln(sqrt(2)*s1) \f$
    real_type gamma_factor;   //!< Constant for evaluating screening functions
    real_type epsilon_factor; //!< Constant for evaluating screening functions
};

struct RelBremIds
{
    //! Model ID
    ModelId model;
    //! ID of an electron
    ParticleId electron;
    //! ID of an positron
    ParticleId positron;
    //! ID of a gamma
    ParticleId gamma;

    //! Whether the IDs are assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model && electron && positron && gamma;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
template<Ownership W, MemSpace M>
struct RelativisticBremData
{
    //! IDs
    RelBremIds ids;

    //! Electron mass [MeVMass]
    units::MevMass electron_mass;

    //! LPM flag
    bool enable_lpm;

    //! LPM table
    using ItemIdT = celeritas::ItemId<unsigned int>;

    template<class T>
    using Items = celeritas::Collection<T, W, M, ItemIdT>;
    Items<RelBremMigdalData> lpm_table;

    //! Element data
    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, ElementId>;
    ElementItems<RelBremElementData> elem_data;

    //! Inverse of the interval for evaluating LPM functions
    static CELER_CONSTEXPR_FUNCTION real_type inv_delta_lpm() { return 100.; }

    //! The upper limit of the LPM variable for evaluating LPM functions
    static CELER_CONSTEXPR_FUNCTION real_type limit_s_lpm() { return 2.0; }

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity() && !lpm_table.empty()
               && !elem_data.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RelativisticBremData& operator=(const RelativisticBremData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        ids           = other.ids;
        electron_mass = other.electron_mass;
        enable_lpm    = other.enable_lpm;
        lpm_table     = other.lpm_table;
        elem_data     = other.elem_data;
        return *this;
    }
};

using RelativisticBremDeviceRef
    = RelativisticBremData<Ownership::const_reference, MemSpace::device>;
using RelativisticBremHostRef
    = RelativisticBremData<Ownership::const_reference, MemSpace::host>;
using RelativisticBremNativeRef
    = RelativisticBremData<Ownership::const_reference, MemSpace::native>;

} // namespace detail
} // namespace celeritas
