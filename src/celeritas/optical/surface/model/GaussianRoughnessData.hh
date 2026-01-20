//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/GaussianRoughnessData.hh
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
 * Storage for Gaussian roughness model data.
 */
template<Ownership W, MemSpace M>
struct GaussianRoughnessData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using SurfaceItems = Collection<T, W, M, SubModelId>;
    //!@}

    //// DATA /////

    SurfaceItems<real_type> sigma_alpha;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !sigma_alpha.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GaussianRoughnessData<W, M>&
    operator=(GaussianRoughnessData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        sigma_alpha = other.sigma_alpha;

        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
