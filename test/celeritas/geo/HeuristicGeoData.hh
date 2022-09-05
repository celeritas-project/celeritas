//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/HeuristicGeoData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/random/RngData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// DATA
//---------------------------------------------------------------------------//
struct HeuristicGeoScalars
{
    // User-configurable options
    Real3     lower{0, 0, 0};
    Real3     upper{0, 0, 0};
    real_type log_min_step{-16.11809565095832}; // 1 nm
    real_type log_max_step{2.302585092994046};  // 10 cm

    // Set from geometry
    VolumeId::size_type num_volumes{};
    bool                ignore_zero_safety{};

    explicit CELER_FUNCTION operator bool() const
    {
        return log_min_step <= log_max_step && num_volumes > 0;
    }
};

template<Ownership W, MemSpace M>
struct HeuristicGeoParamsData
{
    GeoParamsData<W, M> geometry;
    RngParamsData<W, M> rng;
    HeuristicGeoScalars s;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && rng && s;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    HeuristicGeoParamsData&
    operator=(const HeuristicGeoParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        rng      = other.rng;
        s        = other.s;
        return *this;
    }
};

// Special enum to avoid std::vector<bool>
enum LifeStatus : unsigned short
{
    unborn = 0,
    alive,
    dead
};

template<Ownership W, MemSpace M>
struct HeuristicGeoStateData
{
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    GeoStateData<W, M>     geometry;
    RngStateData<W, M>     rng;
    StateItems<LifeStatus> status;

    Collection<real_type, W, M, VolumeId> accum_path;

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return geometry.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && rng && !status.empty() && !accum_path.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    HeuristicGeoStateData& operator=(HeuristicGeoStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry   = other.geometry;
        rng        = other.rng;
        status     = other.status;
        accum_path = other.accum_path;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize and zero-initialize states.
 */
template<MemSpace M>
inline void resize(HeuristicGeoStateData<Ownership::value, M>* state,
                   const HostCRef<HeuristicGeoParamsData>&     params,
                   size_type                                   size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    resize(&state->geometry, params.geometry, size);
    resize(&state->rng, params.rng, size);
    resize(&state->status, size);
    fill(LifeStatus::unborn, &state->status);

    resize(&state->accum_path, params.s.num_volumes);
    fill(real_type{0}, &state->accum_path);
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
