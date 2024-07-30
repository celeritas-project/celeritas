//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/Color.cc
//---------------------------------------------------------------------------//
#include "Color.hh"

#include <iomanip>
#include <sstream>

namespace celeritas
{
//---------------------------------------------------------------------------//
static_assert(static_cast<size_type>(Color::Channel::size_)
              == Color::Byte4{}.size());

//---------------------------------------------------------------------------//
/*!
 * Construct with an \c \#0178ef RGB or RGBA string (100% opaque).
 */
Color Color::from_html(std::string_view rgb)
{
    CELER_VALIDATE((rgb.size() == 7 || rgb.size() == 9) && rgb.front() == '#',
                   << "invalid color string '" << rgb << '\'');

    // Convert base-16 color to an int
    std::string hex_str(rgb.begin() + 1, rgb.end());
    std::size_t count{0};
    auto int_color = std::stol(hex_str, &count, 16);
    CELER_VALIDATE(count == hex_str.size(),
                   << "failed to parse color string '" << rgb << '\'');

    if (hex_str.size() == 8)
        return Color::from_rgba(int_color);
    return Color::from_rgb(int_color);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from an RGB hex value (100% opaque).
 */
Color Color::from_rgb(size_type int_color)
{
    CELER_EXPECT(int_color <= 0xffffffu);
    // Add 100% opacity
    int_color <<= 8;
    int_color |= 0xff;
    return Color(int_color);
}

//---------------------------------------------------------------------------//
/*!
 * Construct from an RGBA value.
 */
Color Color::from_rgba(size_type int_color)
{
    return Color(int_color);
}

//---------------------------------------------------------------------------//
/*!
 * Get representation as an HTML color (\c \#RRGGBB).
 *
 * This drops the opacity
 */
std::string Color::to_html() const
{
    std::ostringstream os;

    int width = 8;
    size_type color = color_;
    if (this->channel(Channel::alpha) == byte_type(0xff))
    {
        // Remove 100% opacity
        color >>= 8;
        width = 6;
    }

    os << '#' << std::setfill('0') << std::hex << std::setw(width) << color;
    return os.str();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
