//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/OnlyReflectionData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Data for only reflections interaction model.
 */
template<Ownership W, MemSpace M>
struct OnlyReflectionData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using SurfaceItems = Collection<T, W, M, SubModelId>;
    //!@}

    SurfaceItems<ReflectionMode> modes;

    //! Whether data are assigned and valid
    explicit CELER_FUNCTION operator bool() const { return !modes.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OnlyReflectionData<W, M>& operator=(OnlyReflectionData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        modes = other.modes;
        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
