//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
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
    real_type el;   //! elastic component
    real_type inel; //! inelastic component
};

//---------------------------------------------------------------------------//
/*!
 * An array of LPM functions, G(s) and \phi(s) to build a table in the range
 * s = [0, limit] with an interval \delta where limit = 2.0 and \delta = 0.01
 * by default.
 */
struct RelBremMigdalData
{
    real_type Gs;   //! LPM G(s)
    real_type phis; //! LPM \phi(s)
};

//---------------------------------------------------------------------------//
/*!
 * A special meta data structure per element used in the differential cross
 * section calculation.
 */
struct RelBremElementData
{
    int       iZ;            //! Atomic number
    real_type logZ;          //! \f$ \ln(Z) \f$
    real_type fZ;            //! \f$ \ln(Z)/3 + f_c (Coulomb correction) \f$
    real_type zFactor1;      //! \f$ ((Fel-fc)+Finel*invZ)\f$
    real_type zFactor2;      //! \f$ (1.0+invZ)/12 \f$
    real_type s1;            //! LPM variables
    real_type inv_logs1;     //! 1/\ln(s1)
    real_type inv_logs2;     //! 1/\ln(sqrt(2)*s1)
    real_type gammaFactor;   //! constants for evaluating screening functions
    real_type epsilonFactor; //!
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

    //! Minimum electron/positron energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy low_energy_limit()
    {
        return units::MevEnergy{1e3}; //! 1 GeV
    }

    //! Maximum electron/positron energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy high_energy_limit()
    {
        return units::MevEnergy{1e8}; //! 100 TeV
    }

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass.value() > 0 && !lpm_table.empty()
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
