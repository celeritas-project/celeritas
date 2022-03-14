//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LPMData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Evaluated LPM data for a single energy gridpoint.
 *
 * These are the LPM suppression functions \f$ G(s) \f$ and \f$ \phi(s) \f$
 * tabulated in the range \f$ s = [0, s_\mathrm{max}] \f$ with an interval \f$
 * \delta \f$ where \f$ s_\mathrm{max} = 2.0 \f$ and \f$ \delta = 0.01 \f$ by
 * default.
 *
 * This is used by \c LPMCalculator.
 */
struct MigdalData
{
    real_type g;   //!< LPM \f$ G(s) \f$
    real_type phi; //!< LPM \f$ \phi(s) \f$
};

//---------------------------------------------------------------------------//
/*!
 * Data needed to calculate the LPM suppression functions.
 */
template<Ownership W, MemSpace M>
struct LPMData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    real_type         electron_mass; //!< Electron mass [MevMass]
    Items<MigdalData> lpm_table;     //!< Tabulated LPM suppression functions

    //// CONSTANTS ////

    //! Inverse of the interval for evaluating LPM functions
    static CELER_CONSTEXPR_FUNCTION real_type inv_delta() { return 100; }

    //! Upper limit of the LPM suppression variable
    static CELER_CONSTEXPR_FUNCTION real_type s_limit() { return 2; }

    //// METHODS ////

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return electron_mass > 0 && !lpm_table.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    LPMData& operator=(const LPMData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        electron_mass = other.electron_mass;
        lpm_table     = other.lpm_table;
        return *this;
    }
};

using LPMDataRef = LPMData<Ownership::const_reference, MemSpace::native>;

} // namespace celeritas
