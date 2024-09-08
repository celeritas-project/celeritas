//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/Color.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Stora an RGBA color.
 *
 * This is used to define volume/material colors used for raytrace rendering.
 * The "byte" is the bit depth of a channel.
 */
class Color
{
  public:
    //!@{
    //! \name Type aliases
    using byte_type = std::uint8_t;
    using size_type = std::uint32_t;
    using Byte4 = Array<byte_type, 4>;
    //!@}

    //! Little-endian indexing for RGBA
    enum class Channel
    {
        alpha,
        blue,
        green,
        red,
        size_
    };

  public:
    // Construct with an \c #0178ef RGB string (100% opaque)
    static Color from_html(std::string_view rgb);

    // Construct from an RGB hex value
    static Color from_rgb(size_type rgb);

    // Construct from an RGBA hex value
    static Color from_rgba(size_type rgba);

    // Construct with transparent black
    Color() = default;

    //! Construct from an integer representation
    explicit CELER_CONSTEXPR_FUNCTION Color(size_type c) : color_{c} {}

    //! Get an integer representation of the color
    explicit CELER_CONSTEXPR_FUNCTION operator size_type() const
    {
        return color_;
    }

    // Get representation as HTML color \#RRGGBB
    std::string to_html() const;

    // Get a single channel
    inline CELER_FUNCTION byte_type channel(Channel c) const;

  private:
    //! Encoded color value
    size_type color_{0};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get a single channel.
 */
CELER_FUNCTION auto Color::channel(Channel c) const -> byte_type
{
    CELER_EXPECT(c < Channel::size_);
    size_type result = color_;
    result >>= (8 * static_cast<size_type>(c));
    result &= 0xffu;
    return static_cast<byte_type>(result);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
