//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/WavelengthShiftData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Material dependent scalar property of wavelength shift (WLS).
 */
struct WlsMaterialRecord
{
    real_type mean_num_photons{};  //!< Mean number of reemitted photons
    real_type time_constant{};  //!< Time delay of WLS [time]

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return mean_num_photons > 0 && time_constant > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Wavelength shift data
 */
template<Ownership W, MemSpace M>
struct WavelengthShiftData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;

    //// MEMBER DATA ////

    OpticalMaterialItems<WlsMaterialRecord> wls_record;

    // Cumulative probability of emission as a function of energy
    OpticalMaterialItems<NonuniformGridRecord> energy_cdf;

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !wls_record.empty() && !energy_cdf.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    WavelengthShiftData& operator=(WavelengthShiftData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        wls_record = other.wls_record;
        energy_cdf = other.energy_cdf;
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
