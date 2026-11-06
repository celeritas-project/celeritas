//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ResponseTallyData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/UniformGridData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
//! Compressed (flattened) index of a {detector, x, y, z, time} tally bin
using ResponseBinId = OpaqueId<struct ResponseBin_>;

//---------------------------------------------------------------------------//
/*!
 * Shared data for tallying detector response on a regular spatial/time grid.
 *
 * The grids are uniform grids of bin \em edges for each of x, y, z, and time,
 * so a grid with \c size edges has <code>size - 1</code> bins. The \c dims
 * array is ordered as <code>{detector, x, y, z, time}</code> and is used to
 * flatten the five-dimensional bin index with a \c HyperslabIndexer (the
 * final dimension changes fastest).
 */
template<Ownership W, MemSpace M>
struct ResponseTallyParamsData
{
    //// DATA ////

    UniformGridData x_grid;
    UniformGridData y_grid;
    UniformGridData z_grid;
    UniformGridData time_grid;
    Array<size_type, 5> dims{};  //!< {detector, x, y, z, time}

    //// METHODS ////

    //! Total number of bins (the size of the flattened tally)
    CELER_FUNCTION size_type num_bins() const
    {
        size_type result = 1;
        for (auto d : dims)
        {
            result *= d;
        }
        return result;
    }

    //! Whether the data are assigned and consistent
    explicit CELER_FUNCTION operator bool() const
    {
        return x_grid && y_grid && z_grid && time_grid && dims[0] > 0
               && dims[1] + 1 == x_grid.size && dims[2] + 1 == y_grid.size
               && dims[3] + 1 == z_grid.size && dims[4] + 1 == time_grid.size;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ResponseTallyParamsData& operator=(
        ResponseTallyParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        x_grid = other.x_grid;
        y_grid = other.y_grid;
        z_grid = other.z_grid;
        time_grid = other.time_grid;
        dims = other.dims;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Per-stream state data for the response tally.
 *
 * \c bin_ids holds each track's \em partial bin ID: when the track is
 * initialized, only the spatial (x, y, z) coordinates of its starting point
 * are filled in, and the detector and time coordinates are zero. The
 * detector and time contributions are added when the track hits a detector.
 * An invalid ID means the track started outside the spatial grid and will
 * not be scored.
 *
 * \c counts is the number of hits in each bin accumulated over the stream's
 * lifetime.
 */
template<Ownership W, MemSpace M>
struct ResponseTallyStateData
{
    //// TYPES ////

    template<class T>
    using BinItems = Collection<T, W, M, ResponseBinId>;
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    StateItems<ResponseBinId> bin_ids;
    BinItems<size_type> counts;

    //// METHODS ////

    //! Number of track slots
    CELER_FUNCTION size_type size() const { return bin_ids.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !bin_ids.empty() && !counts.empty();
    }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    ResponseTallyStateData& operator=(ResponseTallyStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        bin_ids = other.bin_ids;
        counts = other.counts;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
// Resize based on the number of bins and track slots
template<MemSpace M>
void resize(ResponseTallyStateData<Ownership::value, M>* state,
            HostCRef<ResponseTallyParamsData> const& params,
            StreamId,
            size_type size);

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
