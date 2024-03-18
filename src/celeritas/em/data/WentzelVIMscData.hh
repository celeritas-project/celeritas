//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/WentzelVIMscData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/XsGridData.hh"

#include "MscData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Settable parameters and default values for Wentzel VI multiple scattering.
 */
struct WentzelVIMscParameters
{
    using Energy = units::MevEnergy;

    real_type single_scattering_fact{1.25};  //!< single scattering factor
    Energy low_energy_limit{0};
    Energy high_energy_limit{0};

    //! The minimum value of the true path length limit: 1 nm
    static CELER_CONSTEXPR_FUNCTION real_type limit_min_fix()
    {
        return 1e-7 * units::centimeter;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for Wentzel VI MSC.
 */
template<Ownership W, MemSpace M>
struct WentzelVIMscData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    //! Particle IDs
    MscIds ids;
    //! Mass of of electron in MeV
    units::MevMass electron_mass;
    //! User-assignable options
    WentzelVIMscParameters params;
    //! Scaled xs data
    Items<XsGridData> xs;  //!< [mat][particle]

    // Backend storage
    Items<real_type> reals;

    //// METHODS ////

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity() && !xs.empty()
               && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    WentzelVIMscData& operator=(WentzelVIMscData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        electron_mass = other.electron_mass;
        params = other.params;
        xs = other.xs;
        reals = other.reals;
        return *this;
    }

    //! Get the data location for a material + particle
    CELER_FUNCTION ItemId<XsGridData> at(MaterialId mat, ParticleId par) const
    {
        CELER_EXPECT(mat && par);
        size_type result = mat.unchecked_get() * 2;
        result += (par == this->ids.electron ? 0 : 1);
        CELER_ENSURE(result < this->xs.size());
        return ItemId<XsGridData>{result};
    }
};

}  // namespace celeritas
