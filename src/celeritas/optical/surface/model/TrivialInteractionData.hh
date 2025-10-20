//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/TrivialInteractionData.hh
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
 * Data for the trivial surface interaction model.
 */
template<Ownership W, MemSpace M>
struct TrivialInteractionData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using SurfaceItems = Collection<T, W, M, SubModelId>;
    //!@}

    SurfaceItems<TrivialInteractionMode> modes;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const { return !modes.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    TrivialInteractionData<W, M>&
    operator=(TrivialInteractionData<W2, M2> const& other)
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
