//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricInteractionData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//
// Dielectric interface type
enum class DielectricInterface : bool
{
    metal,
    dielectric,
};

//---------------------------------------------------------------------------//
// CLASSES
//---------------------------------------------------------------------------//
/*!
 * Data for the dielectric model denoting which interfaces are
 * dielectric-dielectric and dielectric-metal.
 */
template<Ownership W, MemSpace M>
struct DielectricData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using SurfaceItems = Collection<T, W, M, SubModelId>;
    //!@}

    SurfaceItems<DielectricInterface> interface;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !interface.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DielectricData<W, M>& operator=(DielectricData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        interface = other.interface;
        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Physics grids for the UNIFIED reflection model.
 */
template<Ownership W, MemSpace M>
struct UnifiedReflectionData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using UnifiedModeArray = EnumArray<ReflectionMode, T>;

    template<class T>
    using Items = Collection<T, W, M>;

    using SurfaceGrids = Collection<NonuniformGridRecord, W, M, SubModelId>;
    //!@}

    UnifiedModeArray<SurfaceGrids> reflection_grids;

    //! Backend storage
    Items<real_type> reals;

    //! Number of supported surfaces
    CELER_FUNCTION SubModelId::size_type size() const
    {
        return reflection_grids.front().size();
    }

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return this->size() > 0
               && all_of(reflection_grids.begin(),
                         reflection_grids.end(),
                         [this](auto const& grid) {
                             return grid.size() == this->size();
                         });
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    UnifiedReflectionData<W, M>&
    operator=(UnifiedReflectionData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        for (auto mode : range(ReflectionMode::size_))
        {
            reflection_grids[mode] = other.reflection_grids[mode];
        }
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
