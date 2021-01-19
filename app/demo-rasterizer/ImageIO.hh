//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ImageIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "base/Assert.hh"
#include "base/Array.hh"
#include "base/Types.hh"

namespace demo_rasterizer
{
class ImageStore;
//---------------------------------------------------------------------------//
//! Image construction arguments
struct ImageRunArgs
{
    celeritas::Real3 lower_left;
    celeritas::Real3 upper_right;
    celeritas::Real3 rightward_ax;
    unsigned int     vertical_pixels;
};

void to_json(nlohmann::json& j, const ImageRunArgs& value);
void from_json(const nlohmann::json& j, ImageRunArgs& value);

void to_json(nlohmann::json& j, const ImageStore& value);
//---------------------------------------------------------------------------//
} // namespace demo_rasterizer

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class T, std::size_t N>
void from_json(const nlohmann::json& j, Array<T, N>& value)
{
    CELER_EXPECT(j.size() == N);
    for (std::size_t i = 0; i != N; ++i)
    {
        value[i] = j[i].get<T>();
    }
}

//---------------------------------------------------------------------------//
template<class T, std::size_t N>
void to_json(nlohmann::json& j, const Array<T, N>& value)
{
    j = nlohmann::json::array();
    for (std::size_t i = 0; i != N; ++i)
    {
        j.push_back(value[i]);
    }
}

//---------------------------------------------------------------------------//
} // namespace celeritas
