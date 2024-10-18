//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ParticleData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Storage for dynamic particle data.
 */
template<Ownership W, MemSpace M>
struct ParticleStateData
{
    //// TYPES ////

    using Real3 = Array<real_type, 3>;
    template<class T>
    using Items = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    Items<real_type> energy;  //! Kinetic energy [MeV]
    Items<Real3> polarization;

    //// METHODS ////

    //! Check whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !energy.empty() && !polarization.empty();
    }

    //! State size
    CELER_FUNCTION size_type size() const { return energy.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ParticleStateData& operator=(ParticleStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        energy = other.energy;
        polarization = other.polarization;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize particle states.
 */
template<MemSpace M>
inline void resize(ParticleStateData<Ownership::value, M>* data, size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&data->energy, size);
    resize(&data->polarization, size);
    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
