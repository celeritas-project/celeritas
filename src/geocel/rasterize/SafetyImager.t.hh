//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/SafetyImager.t.hh
//---------------------------------------------------------------------------//
#pragma once

#include "SafetyImager.hh"

#include <fstream>
#include <nlohmann/json.hpp>

#include "ImageIO.json.hh"

#include "detail/SafetyCalculator.hh"

namespace celeritas
{
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
    for (auto i : range(scalars.dims[0]))
    {
        line.clear();
        for (auto j : range(scalars.dims[1]))
        {
            line.push_back(calc_safety(j, i));  // Note: col is 'x' position
        }
        out << nlohmann::json(line).dump() << std::endl;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
