//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurData.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//

enum class GliSurFinishType
{
    polished,
    ground,
    size_
};

enum class GliSurInterfaceType
{
    dielectric_metal,
    dielectric_dielectric,
    size_
};

struct GliSurScalars
{
    ActionId trivial_normal_action;
    ActionId glisur_polished_normal_action;

    ActionId grid_reflectivity_action;

    ActionId glisur_dielectric_interaction;
    ActionId glisur_metal_interaction;

    explicit CELER_FUNCTION operator bool() const
    {
        return trivial_normal_action && glisur_polished_normal_action
               && grid_reflectivity_action && glisur_dielectric_interaction
               && glisur_metal_interaction;
    }
};

template<Ownership W, MemSpace M>
struct GliSurPolishedNormalData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ModelItems = Collection<T, W, M, SurfaceModelId>;
    using SurfacePolishSubset = ItemMap<PerModelSurfaceId, OpaqueId<real_type>>;
    //!@}

    ModelItems<SurfacePolishSubset> polish_table;
    Items<real_type> polish;

    explicit CELER_FUNCTION operator bool() const { return false; }

    template<Ownership W2, MemSpace M2>
    GliSurData<W, M>& CELER_FUNCTION operator=(GliSurData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    GliSurData ...;
   \endcode
 */
template<Ownership W, MemSpace M>
struct GliSurData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using SurfaceItems = Collection<T, W, M, SurfaceId>;
    //!@}

    GliSurScalars scalars;

    SurfaceItems<GliSurFinishType> finish;
    SurfaceItems<GliSurInterfaceType> interface_type;

    explicit CELER_FUNCTION operator bool() const { return false; }

    template<Ownership W2, MemSpace M2>
    GliSurData<W, M>& CELER_FUNCTION operator=(GliSurData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
