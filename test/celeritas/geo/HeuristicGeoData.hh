//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/HeuristicGeoData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/random/data/RngData.hh"
#include "geocel/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/geo/GeoData.hh"

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
    Real3 lower{0, 0, 0};
    Real3 upper{0, 0, 0};
    real_type log_min_step{-16.11809565095832};  // 1 nm (1e-7)
    real_type log_max_step{2.302585092994046};  // 10 cm (1e2)

    static constexpr real_type safety_tol = 0.01;
    // High limit prevents truncation to safety distance
    real_type geom_limit = 5e-8 * units::millimeter;

    // Set from geometry
    VolumeId::size_type num_volumes{};
    bool ignore_zero_safety{};
    VolumeId world_volume;

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
    operator=(HeuristicGeoParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        rng = other.rng;
        s = other.s;
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

    GeoStateData<W, M> geometry;
    RngStateData<W, M> rng;
    StateItems<LifeStatus> status;
    size_type step{0};

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
        geometry = other.geometry;
        rng = other.rng;
        status = other.status;
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
                   HostCRef<HeuristicGeoParamsData> const& params,
                   size_type size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    resize(&state->geometry, params.geometry, size);
    resize(&state->rng, params.rng, StreamId{0}, size);
    resize(&state->status, size);
    fill(LifeStatus::unborn, &state->status);

    resize(&state->accum_path, params.s.num_volumes);
    fill(real_type{0}, &state->accum_path);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
