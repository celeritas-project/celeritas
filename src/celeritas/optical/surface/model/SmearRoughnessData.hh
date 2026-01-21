//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/SmearRoughnessData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Storage for uniform smear roughness model data.
 */
template<Ownership W, MemSpace M>
struct SmearRoughnessData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using SurfaceItems = Collection<T, W, M, SubModelId>;
    //!@}

    //// DATA /////

    SurfaceItems<real_type> roughness;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !roughness.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SmearRoughnessData<W, M>& operator=(SmearRoughnessData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        roughness = other.roughness;

        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
