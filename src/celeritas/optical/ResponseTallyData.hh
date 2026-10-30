//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ResponseTallyData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/UniformGridData.hh"

namespace celeritas
{
namespace optical
{
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
 * \c bin_ids has the compressed (flattened) bin ID of the last hit scored by
 * each track slot, or \c ResponseTallyStateData::invalid_bin() if the track
 * did not score. \c counts is the number of hits in each bin accumulated
 * over the stream's lifetime.
 */
template<Ownership W, MemSpace M>
struct ResponseTallyStateData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    StateItems<size_type> bin_ids;
    Items<size_type> counts;

    //// METHODS ////

    //! Bin ID indicating no hit
    static CELER_CONSTEXPR_FUNCTION size_type invalid_bin()
    {
        return static_cast<size_type>(-1);
    }

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
