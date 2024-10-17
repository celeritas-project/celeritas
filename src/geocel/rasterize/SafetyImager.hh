//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/SafetyImager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <fstream>
#include <nlohmann/json.hpp>

#include "corecel/data/CollectionStateStore.hh"
#include "geocel/GeoTraits.hh"
#include "geocel/rasterize/Image.hh"
#include "geocel/rasterize/ImageIO.json.hh"

#include "detail/SafetyCalculator.hh"

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
    explicit inline SafetyImager(SPConstGeo geo);

    // Save an image
    inline void operator()(ImageParams const& image, std::string filename);

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
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with geometry and build a single state.
 */
template<class G>
SafetyImager<G>::SafetyImager(SPConstGeo geo) : geo_{std::move(geo)}
{
    CELER_EXPECT(geo_);

    host_state_ = {geo_->host_ref(), 1};
}

//---------------------------------------------------------------------------//
/*!
 * Write an image to a file.
 */
template<class G>
void SafetyImager<G>::operator()(ImageParams const& image, std::string filename)
{
    std::ofstream out{filename, std::ios::out | std::ios::trunc};
    CELER_VALIDATE(out, << "failed to open '" << filename << "'");
    out << nlohmann::json(image).dump() << std::endl;

    auto const& scalars = image.scalars();
    real_type max_distance = celeritas::max(scalars.dims[0], scalars.dims[1])
                             * scalars.pixel_width;

    detail::SafetyCalculator calc_safety{
        GeoTrackView{geo_->host_ref(), host_state_.ref(), TrackSlotId{0}},
        image.host_ref(),
        max_distance};

    std::vector<double> line;
    for (auto j : range(scalars.dims[1]))
    {
        line.clear();
        for (auto i : range(scalars.dims[0]))
        {
            line.push_back(calc_safety(i, j));
        }
        out << nlohmann::json(line).dump() << std::endl;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
