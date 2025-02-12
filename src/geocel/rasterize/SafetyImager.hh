//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/SafetyImager.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionStateStore.hh"
#include "geocel/GeoTraits.hh"
#include "geocel/rasterize/Image.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write safety distances from a geometry.
 *
 * The file format is JSON lines:
 * - first line: metadata
 * - each further line: progressive y coordinates
 *
 * \note This is a very rough-and-ready class that should be restructured and
 * integrated with the ray tracer so that it can be executed in parallel on
 * GPU. The interface will change and this will be added to the \c celer-geo
 * app someday!
 */
template<class G>
class SafetyImager
{
    static_assert(std::is_base_of_v<GeoParamsInterface, G>);

  public:
    //!@{
    //! \name Type aliases
    using SPConstGeo = std::shared_ptr<G const>;
    //!@}

  public:
    // Construct with geometry
    explicit SafetyImager(SPConstGeo geo);

    // Save an image
    void operator()(ImageParams const& image, std::string filename);

  private:
    using TraitsT = GeoTraits<G>;
    template<Ownership W, MemSpace M>
    using StateData = typename TraitsT::template StateData<W, M>;
    using HostStateStore = CollectionStateStore<StateData, MemSpace::host>;
    using GeoTrackView = typename TraitsT::TrackView;

    SPConstGeo geo_;
    HostStateStore host_state_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
