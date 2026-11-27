//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/WoodcockData.hh
//! \sa celeritas/woodcock/Woodcock.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/UniformGridData.hh"
#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Index of a volume in which Woodcock tracking is performed
using WoodcockRootId = OpaqueId<struct WoodcockRoot_>;

//---------------------------------------------------------------------------//
/*!
 * Scalar options for Woodcock tracking.
 */
struct WoodcockScalars
{
    using Energy = units::MevEnergy;

    //! Particle type to track
    ParticleId gamma;
    //! Lowest energy for Woodcock tracking
    Energy min_energy{};
    //! Highest energy for which the majorant is tabulated
    Energy max_energy{};
    //! Maximum number of flights per step
    size_type max_flights{};

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return gamma && min_energy > zero_quantity() && max_energy > min_energy
               && max_flights > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent data for Woodcock tracking.
 *
 * A "root" is a volume that encloses a tracking region: see \c WoodcockParams
 * . The data will be empty if no Woodcock tracking is used.
 */
template<Ownership W, MemSpace M>
struct WoodcockParamsData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using VolumeItems = Collection<T, W, M, VolumeId>;
    template<class T>
    using RootItems = Collection<T, W, M, WoodcockRootId>;

    //// DATA ////

    //! Scalar options
    WoodcockScalars scalars;
    //! Tracking root for a volume, null if not a root
    VolumeItems<WoodcockRootId> roots;
    //! Nonzero if the volume is a root or can be inside of one
    VolumeItems<unsigned char> inside;
    //! Majorant cross section as a function of log energy [1/len]
    RootItems<UniformGridRecord> majorants;
    //! Backend storage
    Items<real_type> reals;

    //// METHODS ////

    //! True if Woodcock tracking is enabled
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !roots.empty() && inside.size() == roots.size()
               && !majorants.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    WoodcockParamsData& operator=(WoodcockParamsData<W2, M2> const& other)
    {
        scalars = other.scalars;
        roots = other.roots;
        inside = other.inside;
        majorants = other.majorants;
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
